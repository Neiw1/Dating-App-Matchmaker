import json
import pandas as pd
def parse_filter_preferences(row):
    try:
        return json.loads(row.replace("'", "\""))
    except:
        return {}

def parse_list(val):
    try:
        return json.loads(val.replace("'", "\""))
    except:
        return []
    
def generate_dummies(df, col_name, goal_mapping, train_data_col):
    if col_name == "faculty":
        train_col_name = "faculties"
        df[col_name] = df[col_name].str.upper().fillna('UNKNOWN')
    if col_name == "looking_for":
        train_col_name = "goal"
        df[col_name] = df[col_name].map(goal_mapping).fillna(df['looking_for'])
    dummies = pd.get_dummies(df[col_name], prefix=train_col_name)
    all = [x for x in train_data_col if train_col_name in x]
    dummies = dummies.reindex(columns=all, fill_value=0)
    dummies = dummies.astype(int)
    return dummies

def generate_interests_dummies(df):
    df['interests'] = df['interests'].astype(str)

        # Create interests_Male
    df['interests_Male'] = df['interests'].apply(lambda x: 1 if ('Male' in x or 'Interested in all genders' in x) else 0)

        # Create interests_Female
    df['interests_Female'] = df['interests'].apply(lambda x: 1 if ('Female' in x or 'Interested in all genders' in x) else 0)

    interest_dummies = df[['interests_Male', 'interests_Female']]
    
    return interest_dummies

def map_hobbies(hobby_list, hobbies_map):
    return [hobbies_map.get(hobby, hobby) for hobby in hobby_list]

def map_personality(personality_list, personality_map):
    return [personality_map.get(personality, personality) for personality in personality_list]

def map_values(values_list, values_map):
    return [values_map.get(value, value) for value in values_list]

def get_model_path_MF(model, epoch, lr, wd):
    model_path = f"model_name{model.name}_lr{lr}_wd{wd}_epoch{epoch}"
    return model_path