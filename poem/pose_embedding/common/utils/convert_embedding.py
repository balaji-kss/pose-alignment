"""
Convert EmbeddingDict Frame by Frame format to Person by Person format
"""
from pose_embedding.common import loss_utils_numpy, constants

def convert_embedding_pbp_fbf(embeddingData):
    for sub_id in embeddingData:
        raw_a, raw_b = loss_utils_numpy.get_raw_sigmoid_parameters(embeddingData[sub_id]["metadata"]["sigmoid_a"], embeddingData[sub_id]["metadata"]["sigmoid_b"])
        embeddingData[sub_id]["metadata"].update(dict(raw_a=raw_a, raw_b=raw_b))

def convert_embedding_fbf_pbp(embeddingData):
    # number of person
    # max_person_id = 0
    metadata = embeddingData["meta_info"]
    sigmoid_a, sigmoid_b = loss_utils_numpy.get_sigmoid_parameters(    
        raw_a=metadata["raw_a"],
        raw_b=metadata["raw_b"],
        a_range=(None, constants.sigmoid_a_max)
    )
    metadata["sigmoid_a"] = sigmoid_a
    metadata["sigmoid_b"] = sigmoid_b
    embeddingSubDict = dict()
    for idx, frame_embs in enumerate(embeddingData["embeddings"]):
        # max_person_id = max([em['track_id'] for em in frame_embs], max_person_id)
        updated_subids = []
        for emb in frame_embs:
            if emb["track_id"] not in embeddingSubDict:
                embeddingSubDict[emb["track_id"]] = []
                for i in range(idx):
                    embeddingSubDict[emb["track_id"]].append([])
            embeddingSubDict[emb["track_id"]].append(emb)
            updated_subids.append(emb["track_id"])
        # Not all "track_id" in embeddingSubDict are updated, we should append "[]" to these "track_id"s
        for sub_id in embeddingSubDict:
            if sub_id not in updated_subids:
                embeddingSubDict[sub_id].append([])
        # if len(frame_embs) == 0:
        #     # current frame does not have any embeddings
        #     for sub_id in embeddingSubDict:
        #         embeddingSubDict[sub_id].append([])
    embeddings = dict()
    for sub_id in embeddingSubDict:
        embeddings[sub_id] = dict(data=embeddingSubDict[sub_id], metadata=metadata)
    return embeddings