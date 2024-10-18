import os
import csv
import pandas as pd
import openai
import time

data_path = './data'

valid_news_file = os.path.join(data_path, 'valid', r'news.tsv')
valid_behaviors_file = os.path.join(data_path, 'valid', r'behaviors.tsv')

df_valid_news_data = pd.read_csv(valid_news_file, sep='\t', header=None)
column_names = ['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities']
df_valid_news_data.columns = column_names

df_valid_behaviors_data = pd.read_csv(valid_behaviors_file, sep='\t', header=None)
column_names = ['base_id', 'user_id', 'time', 'history', 'candidate_news']
df_valid_behaviors_data.columns = column_names

all_time = 0
a_win = {"accuracy": 0, "satisfaction": 0, "inspiring": 0, "content_quality": 0, "explainability": 0, "positive_impact": 0, "overall": 0}
b_win = {"accuracy": 0, "satisfaction": 0, "inspiring": 0, "content_quality": 0, "explainability": 0, "positive_impact": 0, "overall": 0}
tie = {"accuracy": 0, "satisfaction": 0, "inspiring": 0, "content_quality": 0, "explainability": 0, "positive_impact": 0, "overall": 0}
index_list = ["accuracy", "satisfaction", "inspiring", "content_quality", "explainability", "positive_impact", "overall"]
a_win_num = 0
b_win_num = 0
tie_num = 0

for user_num in range(1000):
    start = time.time()

    prompt = "Considering you are a user of a news recommendation platform. You have recently read the following news articles:"
    history = df_valid_behaviors_data.iloc[user_num]['history']
    if type(history) == float:
        break
    history = history.split(" ")
    history = history[-5:]
    # print(history)
    for news_id in history:
        # news_id = int(news_id)
        news_title = df_valid_news_data[df_valid_news_data['news_id'] == news_id]['title'].values[0]
        prompt += f"'{news_title}';"

    # print(prompt)


    candidate_news = df_valid_behaviors_data.iloc[user_num]['candidate_news'].split(" ")
    # print(candidate_news)

    prediction_file_A = os.path.join(data_path, "prediction_results", r'prediction_nrms.txt')
    prediction_file_B = os.path.join(data_path, "prediction_results", r'prediction_fm.txt')

    rec_results_A = open(prediction_file_A, "r")
    rec_results_B = open(prediction_file_B, "r")

    rec_results_text = "\nThe top 5 recommended news in the list given to you by Recommendation System A are: "
    rec_list_A = rec_results_A.readline().split(" ")
    rec_list_A = rec_list_A[1].split(",")
    # print(rec_list_A)
    for i in range(5):
        if i == 0:
            news_index = rec_list_A[i][1:]
        else:
            news_index = rec_list_A[i]
        if int(news_index) >= len(candidate_news):
            break
        rec_news_id = candidate_news[int(news_index)][:-2]
        # print(rec_news_id)
        news_title = df_valid_news_data[df_valid_news_data['news_id'] == rec_news_id]['title'].values[0]
        rec_results_text += f"\n{news_title}"
    # print(rec_results_text)

    rec_results_text += "\nThe top 5 recommended news in the list given to you by Recommendation System B are: "
    rec_list_B = rec_results_B.readline().split(" ")
    rec_list_B = rec_list_B[1].split(",")
    # print(rec_list_B)
    for i in range(5):
        if i == 0:
            news_index = rec_list_B[i][1:]
        else:
            news_index = rec_list_B[i]
        if int(news_index) >= len(candidate_news):
            break
        rec_news_id = candidate_news[int(news_index)][:-2]
        # print(rec_news_id)
        news_title = df_valid_news_data[df_valid_news_data['news_id'] == rec_news_id]['title'].values[0]
        rec_results_text += f"\n{news_title}"

    rec_results_A.close()
    rec_results_B.close()

    all_text = prompt + rec_results_text

    # print(all_text)

    all_text = all_text + "\nPlease analyze which recommendation system provides better recommendations based on the following aspects, and provide specific analysis for each aspect. Accuracy: This list of recommendations aligns well with my interests. Satisfaction: I am satisfied with these recommendation results. Inspiring Content: Recommended movies provoke my thoughts, spark my curiosity, encourage further exploration, and enhance my interaction with the recommendation platform. Content Quality: The recommended items are of high quality. Explainability/Transparency: The recommendation is associated with one of my personal information or an interaction history, and it is clear which feature it is. Impact on users: The impact of this recommendation on me is positive."

    all_text = all_text + "Next, based on the results of the above analysis, please evaluate which recommender system performs better overall. Conclude your evaluation with whether 'A wins,' 'B wins,' or 'tie'."

    print(all_text)
    print("=====================================")

    openai.api_base = ""
    openai.api_key = ""
    # client = OpenAI(
    #     api_key="")

    response = openai.ChatCompletion.create(
        messages=[
            {
                "role": "user",
                "content": all_text,
            }
        ],
        model="gpt-4o",
        temperature=0,
    )

    print(response.choices[0].message.content)

    eval_result = response.choices[0].message.content

    # eval_result = eval_result.split("\n")
    # eval_result = [x for x in eval_result if x != '']
    # for e_content in eval_result:
    #     if "Conclusion" in e_content and "A wins" in e_content:
    #         a_win_num += 1
    #     elif "Conclusion" in e_content and "B wins" in e_content:
    #         b_win_num += 1
    #     elif "Conclusion" in e_content and ("tie" in e_content or "Tie" in e_content):
    #         tie_num += 1

    if "A wins" in eval_result:
        a_win_num += 1
    elif "B wins" in eval_result:
        b_win_num += 1
    elif ("tie" in eval_result or "Tie" in eval_result):
        tie_num += 1

    end = time.time()
    print("time: ", end - start)
    all_time += end - start
    print("A wins: {}; B wins: {}; tie: {}".format(a_win_num, b_win_num, tie_num))

print("all time: ", all_time)

print("A wins: ", a_win_num)
print("B wins: ", b_win_num)
print("tie: ", tie_num)

