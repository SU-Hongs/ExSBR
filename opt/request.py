import random, time
from opt.local_llm import LocalLLM

try:
    import openai
    from openai import AzureOpenAI
except ImportError:
    openai = None
    AzureOpenAI = None
import json 
class Request:
    def __init__(self, config):
        self.conf = config
        self.config = config
        self.bad_fp = config.get("bad_log_fp") 
        if config['backend'] == 'local':
            # self.client = LocalLLM(model_dir=config['model_dir'],max_new_tokens=config['max_new_tokens'])
            self.client = LocalLLM(
                model_id=config['model'],    # <- uses --model CLI flag
                ctx_len=config.get("ctx_len", 4096*2),
                max_new_tokens=config['max_new_tokens'],
                batch_size=config.get('bs', 4),
            )
            self.use_local = True
        else:
            if AzureOpenAI is None:
                raise RuntimeError("openai package not installed")
            self.client = AzureOpenAI(
                api_key=config['openai_api_key'],
                api_version=config.get('api_version', '2024-10-21'),
                azure_endpoint=config.get('api_base', '')
            )
            self.use_local = False

    def request(self, user, system=None, message=None):
        if self.use_local:
            return self.client.chat(user, system)
        else:
            return self.openai_request(user, system, message)
        
    def batch_request(self, user_list, system=None):
        if self.use_local:
            return self.client.chat_batch(user_list, system)
        else:
            responses = []
            for u in user_list:
                responses.append(self.openai_request(u, system))
            return responses
    def openai_request(self, user, system=None, message=None):
        '''
        Handles OpenAI communication errors and logs details for empty responses.
        '''
        response_msg = ""
        if system:
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        else:
            messages = [{"role": "user", "content": user}]

        model = self.conf['model']
        engine = self.conf.get('engine', model)  # Use 'engine' if specified, fallback to 'model'

        for delay_secs in (2**x for x in range(0, 10)):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.2,
                    frequency_penalty=0.0

                )
                
                # Check if response contains choices and message
                response_msg = response.choices[0].message.content
                # if response and "choices" in response and len(response.choices) > 0:
                #     response_msg = response.choices[0].message.content
                # else:
                #     response_msg = None

                
                if response_msg is None or len(response_msg) == 0:
                    # Log detailed information if response is empty or None

                    print("Error: Received an empty or None response.")
                    print(f"Model: {model}, Engine: {engine}")
                    print(f"Messages: {messages}")
                    print("Complete Response Object:", response)
                    return ""  # Return None or handle it differently

                break

            except openai.OpenAIError as e:
                # randomness_collision_avoidance = random.randint(0, 1000) / 1000.0

                # sleep_dur = delay_secs + randomness_collision_avoidance

                # print(f"Error: {e}. Retrying in {round(sleep_dur, 2)} seconds.")
                # time.sleep(sleep_dur)
                # ➋ Azure policy block → log & skip further retries
                if ("content_filter" in str(e) or
                    getattr(e, "code", "") == "content_filter"):
                    if self.bad_fp:
                        with open(self.bad_fp, "a", encoding="utf‑8") as f:
                            json.dump({
                                "user"  : user,
                                "system": system or "",
                                "error" : str(e)
                            }, f, ensure_ascii=False)
                            f.write("\n")
                    print("Azure blocked prompt (content_filter). "
                          f"Sample appended to {self.bad_fp}")
                    return ""                # let caller treat as parse fail

                # ➌ other errors → exponential back‑off as before
                sleep_dur = delay_secs + random.random()
                print(f"Error: {e}. Retrying in {round(sleep_dur,2)} s.")
                time.sleep(sleep_dur)
                continue

        return response_msg