
class sentive_sequence_nrn(object):
    """neurone Sequence du modèle de réseau de neurones dynamiques Sentive AI

    Args:
        object (int): numéro donné à l'identifiant du neurone
    """
    def __init__(self, number):
        self.number = number
        self.neuron = {
            "_id":number,
            "schema_version":1,
            "type": "sentive_vision_packed_ratio",
            "layer_id":0,
            "ratio_conn":0, # rapport entre le nombre de connexion de ce neurone et le nb de connexions de ses plus proches voisins
            "DbConnectivity":{
                "pre_synaptique":[],
                "post_synaptique":[],
                "weights":{}
            },
            "meta":{
                "path" : [],
                "tips" : [],
                "nodes" : [],
                "total_length" : 0,
            }
        }