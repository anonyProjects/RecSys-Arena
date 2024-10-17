import pandas as pd
import os
import openai
import time
import argparse

parser = argparse.ArgumentParser(description='Prompt template')
parser.add_argument('--rec_a', required=True, type=str, help='prediction result file of rec a')
parser.add_argument('--rec_b', required=True, type=str, help='prediction result file of rec b')

opt = parser.parse_args()
file_path_rec_a = opt.rec_a
file_path_rec_b = opt.rec_b

train_path = './datasets/ml-1m/processed_data/train_data.csv'
train_data = pd.read_csv(train_path, index_col=0)
test_path = './datasets/ml-1m/processed_data/test_data.csv'
test_data = pd.read_csv(test_path, index_col=0)

age_category = ["under 18 years old", "between 18 and 24 years old", "between 25 and 34 years old", "between 35 and 44 years old", "between 45 and 49 years old", "between 50 and 55 years old", "over 56 years old"]
occupation_category = ["other", "academic/educator", "artist", "clerical/admin", "college/grad student", "customer service", "doctor/health care", "executive/managerial", "farmer", "homemaker", "K-12 student", "lawyer", "programmer", "retired", "sales/marketing", "scientist", "self-employed", "technician/engineer", "tradesman/craftsman", "unemployed", "writer"]
gender_category = ["male", "female"]

all_time = 0
a_win = {"accuracy": 0, "satisfaction": 0, "inspiring": 0, "content_quality": 0, "explainability": 0, "positive_impact": 0, "overall": 0}
b_win = {"accuracy": 0, "satisfaction": 0, "inspiring": 0, "content_quality": 0, "explainability": 0, "positive_impact": 0, "overall": 0}
tie = {"accuracy": 0, "satisfaction": 0, "inspiring": 0, "content_quality": 0, "explainability": 0, "positive_impact": 0, "overall": 0}
index_list = ["accuracy", "satisfaction", "inspiring", "content_quality", "explainability", "positive_impact", "overall"]
a_win_num = 0
b_win_num = 0
tie_num = 0

for u_id in range(1000):
    start = time.time()
    print("user: ", u_id)
    user_list = test_data.query('user_id == @u_id')
    user = user_list.iloc[0]

    prompt = "Considering you are a user of a movie recommendation platform. Your user ID is {user_id}, you are {age}, and your occupation is {occupation}. You are {gender}. ".format(user_id = user["user_id"], age = age_category[user["age"]], occupation=occupation_category[user["occupation"]], gender=gender_category[user["gender"]])

    historical_interactions = train_data.query('user_id == @u_id & label == 1')
    # print(historical_interactions)
    temp_inter_text = "You have recently watched the following movies: "
    for index in range(5):
        cul_inter = historical_interactions.iloc[index]
        temp_inter_text = temp_inter_text + cul_inter["title"] + ", "

        genres = cul_inter["genres"].split('|')
        # print(genres)

        if len(genres)==1:
            temp_inter_text = temp_inter_text + "its genre is " + genres[0] + ". "
        else:
            temp_inter_text = temp_inter_text + "its genre are "
            for g in genres:
                temp_inter_text = temp_inter_text + g + ", "
            temp_inter_text = temp_inter_text[:-2] + ". "


    # print(temp_inter_text)

    rec_results_A = pd.read_csv(file_path_rec_a, index_col=0)
    rec_results_B = pd.read_csv(file_path_rec_b, index_col=0)

    rec_results_text = "The top 5 recommended movies in the list given to you by Recommendation System A are: "
    rec_list_A = []
    rec_results_A = rec_results_A.query('user_id == @u_id')
    for item_id in rec_results_A["item_id"]:
        print(item_id)
        rec_list_A.append(item_id)

        cul_item = train_data.query('item_id == @item_id').iloc[0]

        rec_results_text = rec_results_text + cul_item["title"] + ", "

        genres = cul_item["genres"].split('|')
        # print(genres)

        if len(genres) == 1:
            rec_results_text = rec_results_text + "its genre is " + genres[0] + ". "
        else:
            rec_results_text = rec_results_text + "its genre are "
            for g in genres:
                rec_results_text = rec_results_text + g + ", "
            rec_results_text = rec_results_text[:-2] + ". "

    # print(rec_results_text)

    rec_results_text = rec_results_text + "The top 5 recommended movies in the list given to you by Recommendation System B are: "



    rec_list_B = []
    rec_results_B = rec_results_B.query('user_id == @u_id')
    for item_id in rec_results_B["item_id"]:
        print(item_id)
        rec_list_B.append(item_id)

        cul_item = train_data.query('item_id == @item_id').iloc[0]

        rec_results_text = rec_results_text + cul_item["title"] + ", "

        genres = cul_item["genres"].split('|')
        # print(genres)

        if len(genres) == 1:
            rec_results_text = rec_results_text + "its genre is " + genres[0] + ". "
        else:
            rec_results_text = rec_results_text + "its genre are "
            for g in genres:
                rec_results_text = rec_results_text + g + ", "
            rec_results_text = rec_results_text[:-2] + ". "

    # print(rec_results_text)

    all_text = prompt + temp_inter_text + "The recommendation systems A and B have suggested a list of movies to you based on your personal information and historical interactions. " + rec_results_text

    # print(all_text)

    all_text = all_text + "Please analyze which recommendation system provides better recommendations based on the following aspects, and provide specific analysis for each aspect. Accuracy: This list of recommendations aligns well with my interests. Satisfaction: I am satisfied with these recommendation results. Inspiring Content: Recommended movies provoke my thoughts, spark my curiosity, encourage further exploration, and enhance my interaction with the recommendation platform. Content Quality: The recommended items are of high quality. Explainability/Transparency: The recommendation is associated with one of my personal information or an interaction history, and it is clear which feature it is. Impact on users: The impact of this recommendation on me is positive."
    all_text = all_text + "Next, based on the results of the above analysis, please evaluate which recommender system performs better."
    all_text = all_text + "Conclude your evaluation in this form: Accuracy: 'A wins,' 'B wins,' or 'tie'; Satisfaction: 'A wins,' 'B wins,' or 'tie'; Inspiring Content: 'A wins,' 'B wins,' or 'tie'; Content Quality: 'A wins,' 'B wins,' or 'tie'; Explainability/Transparency: 'A wins,' 'B wins,' or 'tie'; Positive Impact: 'A wins,' 'B wins,' or 'tie'. Overall Winner: 'A wins,' 'B wins,' or 'tie'"

    print(all_text)
    print("=====================================")

    openai.api_base = ""
    openai.api_key = ""

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
    eval_result = eval_result.split("\n")

    eval_result = [x for x in eval_result if x != '']

    print(len(eval_result))

    print(eval_result)

    for e_content in eval_result:
        if "Accuracy" in e_content and "A wins" in e_content:
            a_win["accuracy"] += 1
        elif "Accuracy" in e_content and "B wins" in e_content:
            b_win["accuracy"] += 1
        elif "Accuracy" in e_content and ("tie" in e_content or "Tie" in e_content):
            tie["accuracy"] += 1

        if "Satisfaction" in e_content and "A wins" in e_content:
            a_win["satisfaction"] += 1
        elif "Satisfaction" in e_content and "B wins" in e_content:
            b_win["satisfaction"] += 1
        elif "Satisfaction" in e_content and ("tie" in e_content or "Tie" in e_content):
            tie["satisfaction"] += 1

        if "Inspiring Content" in e_content and "A wins" in e_content:
            a_win["inspiring"] += 1
        elif "Inspiring Content" in e_content and "B wins" in e_content:
            b_win["inspiring"] += 1
        elif "Inspiring Content" in e_content and ("tie" in e_content or "Tie" in e_content):
            tie["inspiring"] += 1

        if "Content Quality" in e_content and "A wins" in e_content:
            a_win["content_quality"] += 1
        elif "Content Quality" in e_content and "B wins" in e_content:
            b_win["content_quality"] += 1
        elif "Content Quality" in e_content and ("tie" in e_content or "Tie" in e_content):
            tie["content_quality"] += 1

        if "Explainability" in e_content and "A wins" in e_content:
            a_win["explainability"] += 1
        elif "Explainability" in e_content and "B wins" in e_content:
            b_win["explainability"] += 1
        elif "Explainability" in e_content and ("tie" in e_content or "Tie" in e_content):
            tie["explainability"] += 1

        if "Positive Impact" in e_content and "A wins" in e_content:
            a_win["positive_impact"] += 1
        elif "Positive Impact" in e_content and "B wins" in e_content:
            b_win["positive_impact"] += 1
        elif "Positive Impact" in e_content and ("tie" in e_content or "Tie" in e_content):
            tie["positive_impact"] += 1

        if "Overall" in e_content and "A wins" in e_content:
            a_win["overall"] += 1
        elif "Overall" in e_content and "B wins" in e_content:
            b_win["overall"] += 1
        elif "Overall" in e_content and ("tie" in e_content or "Tie" in e_content):
            tie["overall"] += 1


    # for i in range(7):
    #     if "A wins" in eval_result[i]:
    #         a_win[index_list[i]] += 1
    #     elif "B wins" in eval_result[i]:
    #         b_win[index_list[i]] += 1
    #     elif "tie" in eval_result[i]:
    #         tie[index_list[i]] += 1
    #
    # if "A wins" in eval_result:
    #     a_win_num += 1
    # elif "B wins" in eval_result:
    #     b_win_num += 1
    # elif ("tie" in eval_result or "Tie" in eval_result):
    #     tie_num += 1

    end = time.time()
    print("time: ", end - start)
    all_time += end - start
    # print("A wins: {}; B wins: {}; tie: {}".format(a_win_num, b_win_num, tie_num))
    print("A wins: ", a_win)
    print("B wins: ", b_win)
    print("tie: ", tie)

print("all time: ", all_time)

print("A wins: ", a_win)
print("B wins: ", b_win)
print("tie: ", tie)
