from cupid.model.data_processor import DataProcessor
from cupid.model.architecture import user_AE, MF
from cupid.model.helper import get_model_path_MF
import pandas as pd
import numpy as np
import torch
class CupidMatchmaker:
    def __init__(self, compatibility_matrix = None):
        AE = user_AE()
        self.model = MF("MF_1V", AE)
        self.model_path = get_model_path_MF(self.model, 26, 0.005, 0.0) 
        self.state = torch.load(self.model_path)
        self.model.load_state_dict(self.state)
        
        if compatibility_matrix is None:
            def compute_compatibility_matrix(tensor_df, model):
                tensor_df = torch.tensor(tensor_df.values, dtype=torch.float32)
                num_users = len(tensor_df)

                # Initialize the compatibility matrix
                compatibility_matrix = np.zeros((num_users, num_users))

                for i in range(num_users):
                    for j in range(num_users):
                        if i != j:  # Skip self-compatibility (or set to 1.0 if desired)
                            # Get user i's data and user j's data
                            user_i_data = tensor_df[i].unsqueeze(0)
                            user_j_data = tensor_df[j].unsqueeze(0)

                            # Create input for model: [user_i_features, user_j_features]
                            compatibility_score = model(user_i_data, user_j_data)

                            compatibility_matrix[i, j] = compatibility_score
                        else:
                            # Self-compatibility - you can set this to 1.0 or 0 based on your needs
                            compatibility_matrix[i, j] = -1
            self.profile_df = pd.read_csv("cupid/model/data/profile.csv")
            self.user_df = pd.read_csv("cupid/model/data/user.csv")
            self.data_processor = DataProcessor(self.profile_df, self.user_df)
            final_df, self.user_id_mapping = self.data_processor.process_data()

            final_df = final_df.drop(columns=['user_id'])

            train_columns = self.data_processor.train_data_col

            train_columns.remove("iid")
            train_columns.remove("pid")
            train_columns.remove("match")

            self.final_df = final_df[train_columns]

            self.compatibility_matrix = compute_compatibility_matrix(self.final_df, self.model)
        else:
            self.compatibility_matrix = compatibility_matrix

    #def add_user():

    def get_partners(self, user_id):
        user_ids = list(self.user_id_mapping)
        if user_id not in user_ids:
            return None
        user_index = user_ids.index(user_id)

        compatibility_scores = self.compatibility_matrix[user_index, :]
        sorted_indices = np.argsort(-compatibility_scores)
        sorted_indices = [i for i in sorted_indices if i != user_index] # Remove self

        top_matches = [user_ids[i] for i in sorted_indices]
        
        return top_matches

    def edit_matrix(self, new_profile_df, new_user_df):
        self.profile_df = pd.concat([self.profile_df, new_profile_df], ignore_index=True)
        self.user_df = pd.concat([self.user_df, new_user_df], ignore_index=True)

        self.data_processor = DataProcessor(self.profile_df, self.user_df)

        final_df, user_id_mapping = self.data_processor.process_data()
        final_df = final_df.drop(columns=['user_id'])
        
        train_columns = self.data_processor.train_data_col.copy()
        self.final_df = final_df[train_columns]

        old_users = self.compatibility_matrix.shape[0]
        total_users = final_df.shape[0]

        new_matrix = np.full((total_users, total_users), -1.0)
        new_matrix[0:old_users, 0:old_users] = self.compatibility_matrix

        tensor_df = torch.tensor(self.final_df.values, dtype=torch.float32)

        for i in range(total_users):
            for j in range(total_users):
                if i != j and new_matrix[i, j] == -1.0:  # Skip self-compatibility (or set to 1.0 if desired)
                    # Get user i's data and user j's data
                    user_i_data = tensor_df[i].unsqueeze(0)
                    user_j_data = tensor_df[j].unsqueeze(0)

                    # Create input for model: [user_i_features, user_j_features]
                    compatibility_score = self.model(user_i_data, user_j_data)

                    new_matrix[i, j] = compatibility_score

        self.compatibility_matrix = new_matrix
        self.user_id_mapping = list(user_id_mapping)