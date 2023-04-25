from sentence_transformers import SentenceTransformer, util
import torch


# make a class for sentece similarity
class SentenceSimilarity:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

        # check device
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

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
