from cupid.model.helper import *
import pandas as pd

class DataProcessor:
    def __init__(self, profile_df, user_df):
        self.profile_df = profile_df
        self.user_df = user_df

        self.goal_mapping = {
            'Long-term partner': 'Long-term Partner',
            'Casual but long-term OK': 'Long-term/Casual',
            'Long-term but casual OK': 'Long-term/Casual'
        }

        self.hobbies_map = {
            "Matcha": "Cafe",
            "Movies & Shows": "Movies & Show",
            "Concert" : "Movies & Show",
            "Coffee" : "Cafe",
            "Bar": "Grabbing a drink",
            "Games": "Video Games"
        }

        self.personality_map = {
            "Curious": "Creative",
            "Positive Energy" : "Cheerful"
        }

        self.values_map = {
            "Personal Growth": "Self-Growth",
            "Honesty and trust" : "Honesty and Trust",
            "Respect": "Honesty and Trust"
        }

        self.gender_mapping = {
            "male": "Male",
            "female": "Female"
        }

        self.train_data_col = ['faculties_FACULTY OF LAW', 'faculties_FACULTY OF SCIENCE',
       'faculties_FACULTY OF POLITICAL SCIENCE',
       'faculties_SASIN GRADUATE INSTITUTE OF BUSINESS ADMINISTION',
       'faculties_FACULTY OF COMMERCE AND ACCOUNTANCY',
       'faculties_FACULTY OF ECONOMICS', 'faculties_FACULTY OF ENGINEERING',
       'faculties_FACULTY OF EDUCATION', 'faculties_FACULTY OF PSYCHOLOGY',
       'faculties_COLLEGE OF PUBLIC HEALTH SCIENCES', 'faculties_UNKNOWN',
       'faculties_FACULTY OF DENTISTRY',
       'faculties_FACULTY OF ALLIED HEALTH SCIENCES',
       'faculties_FACULTY OF PHARMACEUTICAL SCIENCES',
       'faculties_FACULTY OF VETERINARY SCIENCE',
       'faculties_FACULTY OF NURSING', 'faculties_FACULTY OF MEDICINE',
       'faculties_FACULTY OF ARTS',
       'faculties_THE SIRINDHORN THAI LANGUAGE INSTITUTE',
       'faculties_FACULTY OF COMMUNICATION ARTS',
       'faculties_LANGUAGE INSTITUTE',
       'faculties_FACULTY OF FINE AND APPLIED ARTS',
       'faculties_FACULTY OF ARCHITECTURE',
       'faculties_SCHOOL OF INTEGRATED INNOVATION',
       'faculties_SCHOOL OF AGRICULTURAL RESOURCES',
       'faculties_FACULTY OF SPORTS SCIENCE', 'faculties_GRADUATE SCHOOL',
       'faculties_COLLEGE OF POPULATION STUDIES', 'goal_Long-term Partner',
       'goal_Long-term/Casual', 'goal_Casual Fun', 'goal_New Friends',
       'goal_Still figuring it out', 'goal_To say I did it', 'hobbies_Travel',
       'hobbies_Movies & Show', 'hobbies_Photography',
       'hobbies_Grabbing a drink', 'hobbies_Music', 'hobbies_Sports',
       'hobbies_Gym', 'hobbies_Cafe', 'hobbies_Nature', 'hobbies_Reading',
       'hobbies_Video Games', 'hobbies_Food', 'hobbies_Pets', 'hobbies_Art',
       'hobbies_Cooking', 'hobbies_Dancing', 'personality_Cheerful',
       'personality_Reliable', 'personality_Active', 'personality_Caring',
       'personality_Introverted', 'personality_Organized',
       'personality_Creative', 'personality_Thoughtful',
       'personality_Confident', 'personality_Playful', 'personality_Ambitious',
       'personality_Calm', 'personality_Chill', 'values_Honesty and Trust',
       'values_Self-Growth', 'values_Having fun',
       'values_Independence and Balance', 'values_Meaningful Conversation',
       'values_Emotional Support and Empathy', 'values_Physical Affection',
       'gender_Male', 'gender_Female', 'age', 'interests_Male',
       'interests_Female', 'iid', 'pid', 'match']
        
        all_hobbies = [x for x in self.train_data_col if 'hobbies' in x]
        self.all_hobbies = [hobby.split("_")[1] for hobby in all_hobbies]

        all_personality = [x for x in self.train_data_col if 'personality' in x]
        self.all_personality = [personality.split("_")[1] for personality in all_personality]

        all_values = [x for x in self.train_data_col if 'values' in x]
        self.all_values = [value.split("_")[1] for value in all_values]

    def process_data(self):
        self.profile_df['filter_preferences'] = self.profile_df['filter_preferences'].apply(parse_filter_preferences)
        self.profile_df['hobbies'] = self.profile_df['filter_preferences'].apply(lambda d: d.get("1", []))
        self.profile_df['personality'] = self.profile_df['filter_preferences'].apply(lambda d: d.get("2", []))
        self.profile_df['values'] = self.profile_df['filter_preferences'].apply(lambda d: d.get("3", []))

        self.profile_df['interests'] = self.profile_df['interests'].apply(parse_list)

        faculty_dummies = generate_dummies(self.profile_df, 'faculty', self.goal_mapping, self.train_data_col)

        goal_dummies = generate_dummies(self.profile_df, "looking_for", self.goal_mapping, self.train_data_col)

        interests_dummies = generate_interests_dummies(self.profile_df)

        self.profile_df['hobbies'] = self.profile_df['hobbies'].apply(lambda x: map_hobbies(x, self.hobbies_map))

        filter_preferences = pd.DataFrame()

        for hobby in self.all_hobbies:
            filter_preferences[f'hobbies_{hobby}'] = self.profile_df['hobbies'].apply(lambda x: int(hobby in x))

        self.profile_df['personality'] = self.profile_df['personality'].apply(lambda x: map_personality(x, self.personality_map))
        for personality in self.all_personality:
            filter_preferences[f'personality_{personality}'] = self.profile_df['personality'].apply(lambda x: int(personality in x))

        self.profile_df['values'] = self.profile_df['values'].apply(lambda x: map_values(x, self.values_map))
        for values in self.all_values:
            filter_preferences[f'values_{values}'] = self.profile_df['values'].apply(lambda x: int(values in x))

        self.profile_df = self.profile_df.merge(self.user_df[['user_id', 'age', 'gender']], on='user_id', how='left')
        self.profile_df['gender'] = self.profile_df['gender'].map(self.gender_mapping)

        gender_dummies = pd.get_dummies(self.profile_df['gender'], prefix="gender")
        gender_dummies = gender_dummies.astype(int)

        final_df = pd.concat([
            faculty_dummies,
            goal_dummies,
            filter_preferences,
            gender_dummies,
            self.profile_df[['age']],
            interests_dummies,
            self.profile_df[['user_id']]
        ], axis=1)

        return final_df, self.profile_df["user_id"]

