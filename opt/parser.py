import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='SBR Prompting')
    parser.add_argument('--model', 
                        type=str,
                        default='gpt-4o-mini',
                        help='which model as recommender, options: gpt-3.5-turbo')
    parser.add_argument('--seed', 
                        type=int,
                        default=998,
                        help='options: 42, 625, 2023, 0, 10')
    parser.add_argument('--candidate_size', 
                        type=int,
                        default=20,
                        help='options: 10, 20')
    parser.add_argument('--dataset', 
                        type=str,
                        help='use which datset: bundle/games/ml-1m')
    parser.add_argument('--train_num', 
                        type=int,
                        default=50,
                        help='options: 50,150')
    parser.add_argument('--N_t', 
                        type=int,
                        default=32,
                        help='options: 16,32')
    parser.add_argument('--backend', type=str, 
                        default='openai', 
                        help='run with local HF model or OpenAI API')
    parser.add_argument('--bs', type=int, default=4, help='generation batch-size for local backend')
    parser.add_argument('--max_new_tokens',
                        type=int,
                        default=2000,
                        help='maximum number of tokens to generate per call')
    

    args = parser.parse_args()
    
    return args
