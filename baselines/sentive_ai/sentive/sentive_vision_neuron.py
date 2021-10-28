class sentive_vision_neuron(object):
    """
        prototype de données des objets neurones vision segment arcs
        Cette classe ne contient aucune fonction mais ne sert qu'à conserver les données
        Les fonctions sont dans une autre classe.
    """
    def __init__(self, number):
        self.number = number
        self.neuron = {
            "_id":number,
            "schema_version":2,
            "type": "sentive_vision_arcs",
            "layer_id":0,
            "ratio_conn":0,
            "DbConnectivity":{
                "pre_synaptique":[],
                "post_synaptique":[],
                "lateral_connexion":[],
                "weights":{}
            },
            "meta":{
                "center":{
                    "x":0,
                    "y":0
                },
                "matrix_width":1
            }
        }