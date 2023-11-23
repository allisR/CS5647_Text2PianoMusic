import openai
import numpy as np
import pandas as pd
import logging


GPT_API_KEY = "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"
# API_ENDPOINT = "https://api.openai.com/v1/chat/completions"

system = 'You are MusicGPT, a chat bot that knows all piano songs. When a user gives you a pair of composer and canonical title of a piano song, you return them 3 sentences briefly describe the key (must include), melody, style and typical chord progressions in that piano song.'

# Get prompts
def getPrompt(df):
    composer = df['canonical_composer']
    title = df['canonical_title']
    # year = df['year']
    prompt = "Composer:" + composer + ";Title:" + title
    return prompt


def getCaption(df):
    prompt = df['prompt']
    print(prompt)
    history = [{'role': 'system', 'content': system}, {'role': 'user', 'content': prompt}]

    #openai request
    print('[*] Making request to OpenAI API')
    openai.api_key = GPT_API_KEY

    try:
        response = openai.ChatCompletion.create(
            model = 'gpt-4-0613',
            messages = history
        )
        caption = response["choices"][0]["message"]["content"]
        # print(caption)

        return caption
    
    except Exception as e:
        logging.error('Failed to upload to ftp: '+ str(e))
        return None



if __name__ == '__main__':
    raw_file = "/Users/allison/MainDoc/TO_DO/text2midi/maestro-v2.0.0.csv"
    raw_df = pd.read_csv(raw_file)
    raw_df['prompt'] = raw_df.apply(getPrompt, axis=1)

    prompt_df = raw_df.drop_duplicates(['prompt', 'split'])
    prompt_df.to_csv(f"/Users/allison/MainDoc/TO_DO/text2midi/clean+prompt.csv", index=False)

    prompt_file = "/Users/allison/MainDoc/TO_DO/text2midi/clean+prompt.csv"
    prompt_df = pd.read_csv(prompt_file)

    for i, df in enumerate(np.array_split(prompt_df, 3)):
        df['caption'] = df.apply(getCaption, axis=1)
        df.to_csv(f"data{i+1}.csv", index=False)

    merge_dfs = []

    # Merge three dfs
    for i in range(1, 4):
        filepath = "./data"+str(i)+".csv"
        df = pd.read_csv(filepath)
        merge_dfs.append(df)

    clean_df = pd.concat(merge_dfs, axis=0)
    clean_df.to_csv(f"./clean+GPTcaption.csv", index=False)