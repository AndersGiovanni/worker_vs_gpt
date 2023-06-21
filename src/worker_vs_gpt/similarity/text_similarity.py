from typing import Dict, List
from sentence_transformers import SentenceTransformer, util
import torch
from worker_vs_gpt.config import MODELS_DIR

MODEL = MODELS_DIR / "intfloat_e5-base"


# make a class for sentece similarity
class SentenceSimilarity:
    def __init__(self, model_name):
        self.model = SentenceTransformer(MODEL, cache_folder=MODELS_DIR)

        # check device
        self.device = get_device()
        self.model = SentenceTransformer(model_name).to(self.device)
        self.sim_matrix = None
        self.features = None
        self.feature_mat = None
        self.R_feature = None
        self.base_embeddings = None
        self.dataset_embeddings = None

    def compute_similarity_individual(
        self, dataset, labels, text_field: str
    ) -> Dict[str, List[float]]:
        """For each base sample, compute the similarity to all the generated samples.

        Returns
        -------
        Dict[str, List[float]]
            Dictionary with cosine similarity scores for each label
        """

        h_text_all = dataset[text_field]
        h_text_unique = list(set(h_text_all))
        augmented_h_text = dataset[f"augmented_{text_field}"]
        target_labels = dataset["target"].numpy()

        # Compute embedding for both lists
        h_text_unique_embeddings = self.model.encode(
            h_text_unique, convert_to_tensor=True
        )
        augmented_h_text_embeddings = self.model.encode(
            augmented_h_text, convert_to_tensor=True
        )

        # Compute cosine-similarities for each sentence with each other sentence
        cosine_scores = util.pytorch_cos_sim(
            h_text_unique_embeddings, augmented_h_text_embeddings
        )

        similarities = {k: {} for k in labels}
        # for each base sample, we compute the similarity to all the augmented samples
        for idx, (text, augmented_text, label) in enumerate(
            zip(h_text_all, augmented_h_text, target_labels)
        ):
            h_text_idx = h_text_unique.index(text)

            if not text in similarities[labels[label]]:
                similarities[labels[label]][text] = [
                    (cosine_scores[h_text_idx, idx].item(), augmented_text)
                ]
            else:
                similarities[labels[label]][text].append(
                    (cosine_scores[h_text_idx, idx].item(), augmented_text)
                )

        return similarities

    def get_mean_similarity(self, s1, s2):
        # Compute embedding for both lists
        embedding_1 = self.model.encode(s1, convert_to_tensor=True)
        embedding_2 = self.model.encode(s2, convert_to_tensor=True)

        sim_score = torch.mean(util.pytorch_cos_sim(embedding_1, embedding_2), dim=1)

        if self.device == "cuda":
            return sim_score.cpu().detach().numpy()
        else:
            return sim_score.detach().numpy()

    def compute_sim_matrix(self, s1, s2):
        # Compute embedding for both lists
        embedding_1 = self.model.encode(s1, convert_to_tensor=True)
        embedding_2 = self.model.encode(s2, convert_to_tensor=True)
        self.sim_matrix = util.pytorch_cos_sim(embedding_1, embedding_2)
        return

    #
    def get_similarity_sources_targets(self, row=True):
        if row:
            # mean of rows of torch tensor sim_matrix
            sim_score = torch.mean(self.sim_matrix, dim=1)
        else:
            sim_score = torch.mean(self.sim_matrix, dim=0)

        if self.device == "cuda":
            return sim_score.cpu().detach().numpy()
        else:
            return sim_score.detach().numpy()

    # method to get the similarity score between source sentence and list of target sentences
    def get_similarity_pairs(self):
        # get diagnol elements
        diag = torch.diag(self.sim_matrix)

        if self.device == "cuda":
            return diag.cpu().detach().numpy()
        else:
            return diag.detach().numpy()

    def prepare_features_labels(self, text):
        # Compute embedding for both lists
        self.features = self.model.encode(text, convert_to_tensor=True)
        return

    def compute_TransRate(self):
        # matrix multiplication of features and transposed of features
        raise NotImplementedError

        self.feature_mat = torch.mul(
            torch.matmul(self.features.T, self.features), 1 / (self.features.shape[0])
        )

        self.feature_mat = torch.add(
            torch.eye(self.features.shape[1]).to(self.device), self.feature_mat
        )
        # determinant of feature matrix
        self.R_feature = torch.det(self.feature_mat)
        # log of determinant
        self.R_feature = torch.log(self.R_feature)
        # multiply by 1/2
        self.R_feature = torch.mul(self.R_feature, 1 / 2)
