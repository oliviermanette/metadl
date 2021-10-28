class sentive_angle_neuron(object):
    """
        prototype de données des objets neurones vision segment arcs
        Cette classe ne contient aucune fonction mais ne sert qu'à conserver les données
        Les fonctions sont dans une autre classe.
    """
    def __init__(self, number):
        self.number = number
        self.neuron = {
            "_id":number,
            "schema_version":1,
            "type": "sentive_angle_neuron",
            "layer_id":0,
            "ratio_conn":0,
            "DbConnectivity":{
                "pre_synaptique":[],
                "post_synaptique":[],
                "lateral_connexion":[],
                "weights":{}
            },
            "meta":{
                "orientation":{
                    "x":0,
                    "y":0
                },
                "matrix_width":3,
                "angle":-99,
                "before_length":-1,
                "after_length":-1
            }
        }