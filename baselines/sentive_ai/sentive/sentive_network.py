import copy

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

import networkx as nx

from .sentive_neuron_helper import sentive_neuron_helper

class sentive_network(object):

    def __init__(self, episode):

        self.episode = episode
        # print("and here")
        plt.matshow(episode[:,:,0])
        
        ###########################################
        # meta parameters
        self.SEUIL = 0.35

        self.IMG_SIZE = 28
        self.angle_tolerance_deg = 17
        self.ANGL_TOL = np.pi * self.angle_tolerance_deg / 180
        self.angle_tolerance_deg = 17
        self.ANGL_TOL2 = np.pi * self.angle_tolerance_deg / 180

        # si plus petit que EPSILON, considère que c'est égal
        self.ANGL_EPSILON = np.pi * 1 / 180
        
        # POURCENTAGE : MERGE_LIM = 75%
        self.MERGE_LIM = 90
        self.limite_merge = 1 - self.MERGE_LIM/100

        # self.MAX_ANGL = 0.75

        # POURCENTAGE DE PIXELS UNIQUE MINIMUM 
        self.MIN_PIXEL = 10

        self.MIN_PATH = 3

        # end metaparameters
        ###########################################

        # nb est le premier identifiant pour créer les neurones
        self.nb = 0
        # liste contenant tous les neurones : pool_vision
        # self.pool_vision = []
        self.nrn_pxl_map = np.zeros([self.IMG_SIZE,self.IMG_SIZE])
        self.nrn_l2_map = np.zeros([self.IMG_SIZE,self.IMG_SIZE])
        self.np_coord = []

        # fonctions utilitaires
        # neuron_tools 
        self.nrn_tls = sentive_neuron_helper()

        self.glbl_prm = {
            "cg":{"x":0,"y":0},
            "u_axis":{"x":0,"y":0},
            "v_axis":{"x":0,"y":0}
            }
        
        self.nb_ltrl_conn = []

        self.nb_nrn_pxls = 0
        self.nrn_pxls = []

        self.nrn_segments = []
        self.slct_sgmts = []
        self.nrn_saccade = []


    def layer_1(self):
        ##################################################
        ######## NEURONES DE LA COUCHE 1 (t_1) #########
        ##################################################
        self.nrn_tls.new_layer()
        # Crée un neurone par pixel au début:
        pxl_coord = []
        for y in range(1,self.IMG_SIZE-1):
            for x in range(1,self.IMG_SIZE-1):
                if self.episode[y][x][0]>self.SEUIL:
                    nb  = self.nrn_tls.add_new_nrn()
                    self.nrn_pxls.append(nb)
                    self.nrn_tls.lst_nrns[nb].neuron["meta"]["center"]["x"] = x
                    self.nrn_tls.lst_nrns[nb].neuron["meta"]["center"]["y"] = y
                    self.nrn_tls.lst_nrns[nb].neuron["meta"]["matrix_width"] = 1
                    
                    self.nrn_pxl_map[y][x] = nb

                    pxl_coord.append([x,y])
        
        self.nb_nrn_pxls = copy.deepcopy(self.nrn_tls.nb_nrns)
        print("nombre de neurones taille 1:",self.nb_nrn_pxls)
        # print("*"*40,"\n")

        pca = PCA(n_components=2)
        pca.fit(pxl_coord)
        # on obtient les résultats ici:
        # print(pca.components_)
        self.glbl_prm["u_axis"]["x"]=pca.components_[0][0]
        self.glbl_prm["u_axis"]["y"]=pca.components_[0][1]
        self.glbl_prm["v_axis"]["x"]=pca.components_[1][0]
        self.glbl_prm["v_axis"]["y"]=pca.components_[1][1]

        self.np_coord = np.array(pxl_coord)
        self.glbl_prm["cg"]["x"] = np.mean(self.np_coord[:,0])
        self.glbl_prm["cg"]["y"] = np.mean(self.np_coord[:,1])

    
    def layer_2(self):
        ##################################################
        ########## NEURONES DE LA COUCHE 2 (t_3) #########
        ##################################################
        # Les neurones de cette couche ont des champs récepteurs 
        # qui sont des matrices de *3x3*
        # avec des mata paramètres les décrivants.
        self.nrn_tls.new_layer()
        
        lst_nrn2_pos = []

        nb_min = 0

        for neuron_idx in range(self.nrn_tls.nb_nrns):
            # position du centre du neurone
            x = self.nrn_tls.lst_nrns[neuron_idx].neuron["meta"]["center"]["x"]
            y = self.nrn_tls.lst_nrns[neuron_idx].neuron["meta"]["center"]["y"]

            # sub_pxl_map contient les identifiants de chaque neurone pixel sur une carte nrnl_map
            sub_pxl_map = self.nrn_pxl_map[y-1:y+2, x-1:x+2]

            # crée un nouveau neurone de taille 3
            nb  = self.nrn_tls.add_new_nrn()
            if nb_min == 0:
                nb_min = nb
            lst_nrn2_pos.append(nb)
            self.nrn_tls.lst_nrns[nb].neuron["meta"]["center"]["x"] = x
            self.nrn_tls.lst_nrns[nb].neuron["meta"]["center"]["y"] = y
            self.nrn_tls.lst_nrns[nb].neuron["meta"]["matrix_width"] = 3
            self.nrn_tls.lst_nrns[nb].neuron["DbConnectivity"]["pre_synaptique"] = list(set(sub_pxl_map.ravel()))

            self.nrn_l2_map[y][x] = self.nrn_tls.lst_nrns[nb].neuron["_id"]

            for i in range(len(self.nrn_tls.lst_nrns[nb].neuron["DbConnectivity"]["pre_synaptique"])-1,-1,-1):
                if self.nrn_tls.lst_nrns[nb].neuron["DbConnectivity"]["pre_synaptique"][i]==0:
                    self.nrn_tls.lst_nrns[nb].neuron["DbConnectivity"]["pre_synaptique"].pop(i)
                else:
                    self.nrn_tls.netGraph.add_edge(self.nrn_tls.lst_nrns[nb].neuron["_id"],self.nrn_tls.lst_nrns[nb].neuron["DbConnectivity"]["pre_synaptique"][i])
                    nrn_pxl = self.nrn_tls.get_neuron_from_id(self.nrn_tls.lst_nrns[nb].neuron["DbConnectivity"]["pre_synaptique"][i])
                    nrn_pxl["DbConnectivity"]["post_synaptique"].append(self.nrn_tls.lst_nrns[nb].neuron["_id"])

        # détermination des connexions latérales
        for nrn_pos in range (nb_min, self.nrn_tls.nb_nrns):
            nrn2 = self.nrn_tls.lst_nrns[nrn_pos].neuron
            # position du centre du neurone
            x = nrn2["meta"]["center"]["x"]
            y = nrn2["meta"]["center"]["y"]

            # sub_pxl_map contient les identifiants de chaque neurone pixel sur une carte nrnl_map
            sub_pxl_map = self.nrn_l2_map[y-1:y+2, x-1:x+2]
            nrn2["DbConnectivity"]["lateral_connexion"] = list(set(sub_pxl_map.ravel().astype(int)))
            for i_pos in range(len(nrn2["DbConnectivity"]["lateral_connexion"])-1,-1,-1):
                if nrn2["DbConnectivity"]["lateral_connexion"][i_pos] == 0:
                    nrn2["DbConnectivity"]["lateral_connexion"].pop(0)
                else:
                    self.nrn_tls.add_edge(nrn2["_id"],nrn2["DbConnectivity"]["lateral_connexion"][i_pos])
                    self.nrn_tls.increment_weight(nrn2,nrn2["DbConnectivity"]["lateral_connexion"][i_pos])

        
        self.nrn_tls.nb_2_1st_layers = len(self.nrn_tls.lst_nrns)
        print("nombre de neurones couche 1 & 2:",self.nrn_tls.nb_2_1st_layers)
        # print("*"*40)


    def find_tips(self, cp_lst_nrns, lthrshld_tip, lthrshld_nod, G, r_thrshld_tip=-1):
        l_tmp_tips = [] # id des neurones situés à une extrémité
        l_tmp_node = [] # id des neurones au carrefour
        l_tmp_stop = []
        liste_electeurs = []
        liste_candidats = []
        see_tips = []
        for pos in range(len(cp_lst_nrns)):
            # nrn = cp_lst_nrns[pos].neuron
            nrn = self.nrn_tls.get_neuron_from_id(cp_lst_nrns[pos])
            # Sélection des neurones TIPS
            see_tips.append(len(nrn["DbConnectivity"]["lateral_connexion"]))
            if len(nrn["DbConnectivity"]["lateral_connexion"])<=lthrshld_tip:
                l_tmp_tips.append(nrn["_id"])
            if nrn["ratio_conn"]<=r_thrshld_tip:
                l_tmp_tips.append(nrn["_id"])
            # Sélection des neurones NODES
            if len(nrn["DbConnectivity"]["lateral_connexion"])>=lthrshld_nod:
                l_tmp_stop.append(nrn["_id"])
                lbl_not_found = True
                for i in range(len(liste_electeurs)):
                    if len(set(liste_electeurs[i]).intersection(set(nrn["DbConnectivity"]["lateral_connexion"])))>0:
                        liste_electeurs[i].extend(nrn["DbConnectivity"]["lateral_connexion"])
                        liste_candidats[i].append(nrn["_id"])
                        lbl_not_found = False
                if lbl_not_found:
                    liste_electeurs.append(nrn["DbConnectivity"]["lateral_connexion"])
                    liste_candidats.append([nrn["_id"]])
        # print("seuil TIPS",lthrshld_tip)
        # print("see_tips\n",see_tips)
        # regroupement des nodes en trop grand nombres
        vainqueurs_elections = []
        # déroulement du scrutin
        for scrutin in range(len(liste_candidats)):
            liste_votants = list(set(liste_electeurs[scrutin]))
            resultats_election_locale = np.zeros(len(liste_candidats[scrutin]))
            # chaque électeur procède à son vote pour les candidats locaux liste_candidats[scrutin]
            for votant_id in range(len(liste_votants)):
                votant = self.nrn_tls.get_neuron_from_id(liste_votants[votant_id])
                if votant != '': 
                    vote = votant["DbConnectivity"]["weights"]
                    for candidat_id in range(len(liste_candidats[scrutin])):
                        try:
                            resultats_election_locale[candidat_id] += vote[liste_candidats[scrutin][candidat_id]]
                        except:
                            continue
            vainqueurs_elections.append(liste_candidats[scrutin][np.argmax(resultats_election_locale)])   
            # print("resultats_election_locale",resultats_election_locale,"candidats:",liste_candidats[scrutin],"vainqueur:",vainqueurs_elections[len(vainqueurs_elections)-1])
        l_tmp_node = vainqueurs_elections
        l_tmp_tips = list(set(l_tmp_tips))
        # longuest = []
        # for t in l_tmp_tips:
        #     tmp_path_length = []
        #     for n in l_tmp_node:
        #         tmp_path_length.append(int(nx.shortest_path_length(G,t,n)))
        #     longuest.append(min(tmp_path_length))
        # # réarrange les extrémités en fonction de la longueur des chemins les plus courts
        # l_tmp_tips = np.array(l_tmp_tips)
        # l_tmp_tips = l_tmp_tips[np.flip(np.argpartition(np.array(longuest),len(longuest)-1))]
        return l_tmp_tips, l_tmp_node, np.array(l_tmp_stop)

    
    def get_nrn_from_path(self, list_path_nrn_id):
        """A partir de la liste des neurones passés en paramètre,
            retourne l'ensemble des connexions latérales mobilisés
        Args:
            list_path_nrn_id (list): nrn_id
        """
        lst_output = []
        for nrn2_id in list_path_nrn_id:
            nrn2 = self.nrn_tls.get_neuron_from_id(nrn2_id)
            lst_output.extend(nrn2["DbConnectivity"]["lateral_connexion"])
        
        return set(lst_output)


    def layer_3(self):
        """ Détermine * les neurones séquences *
        """
        # création d'une nouvelle couche 
        self.nrn_tls.new_layer()
        
        # lien vers la 2eme couche:
        G = self.nrn_tls.layer_graph[1]
        # G = self.nrn_tls.netGraph
        G2 = copy.deepcopy(G)

        pos_2_cut_off = {
                "connector":[],
                "position":[],
                "nrn_id":[],
                "crnt_nrn":[]
            }

        bl_got_segmented = False

        # copie de la liste des pixels
        # cp_lst_pxls = copy.deepcopy(self.nrn_pxls)

        cp_lst_nrns = [] # copie de la liste des neurones de la 2eme couche
        # calcul des ratios pour chaque neurone
        nrn_ratio_conn = []
        nb_ltrl_conn = []
        for nrn_pos in self.nrn_tls.lst_nrns:
            nrn = nrn_pos.neuron
            if nrn["layer_id"]==2:
                cp_lst_nrns.append(nrn["_id"])
                nb_ltrl_conn.append(len(nrn["DbConnectivity"]["lateral_connexion"]))
                tmp_num_conn = []                
                for nrn_lat in nrn["DbConnectivity"]["lateral_connexion"]:
                    tmp_num_conn.append(len(self.nrn_tls.lst_nrns[nrn_lat-1].neuron["DbConnectivity"]["lateral_connexion"]))
                new_ratio = len(nrn["DbConnectivity"]["lateral_connexion"])/np.mean(tmp_num_conn)
                nrn["ratio_conn"] = new_ratio
                nrn_ratio_conn.append(new_ratio)

        # seuil nombre relatif de connexions
        # r_thrshld_tip = min(nrn_ratio_conn)
        lthrshld_tip = min(nb_ltrl_conn)+1 # seuil pour détecter les extrémités
        lthrshld_nod = max(nb_ltrl_conn) # seuil pour détecter les nœuds


        # Sélectionner plusieurs tips potentiels dans la liste.
        l_tmp_tips, nrn_nodes_id, nrn_stop_id = self.find_tips(cp_lst_nrns, lthrshld_tip, lthrshld_nod, G)
        # print("neurons TIPS:",l_tmp_tips,", nrn NODES",nrn_nodes_id, "et neurons STOP:",nrn_stop_id)
        # initial_tips = copy.deepcopy(l_tmp_tips)
        if len(l_tmp_tips)<=1:
            l_tmp_tips.extend(nrn_stop_id)

        nb_max = len(cp_lst_nrns)
        int_limit = 10
        reste_percent = 100
        nb_min = -1
        print("taille neurones à séquencer :", nb_max)

        while reste_percent>=self.MIN_PATH and int_limit>=0:
            # Calculer les distances neuronales entre chacun
            tip_max_length = 0
            tip_1 = -1
            tip_2 = -1
            for pos_tp_1 in range(len(l_tmp_tips)-1):
                for pos_tp_2 in range(pos_tp_1+1, len(l_tmp_tips)):
                    try:
                        tmp_max_length = nx.shortest_path_length(G2, source=l_tmp_tips[pos_tp_1], target=l_tmp_tips[pos_tp_2])
                    except:
                        tmp_max_length = 0
                    if tip_max_length<tmp_max_length:
                        tip_max_length = tmp_max_length
                        tip_1 = l_tmp_tips[pos_tp_1]
                        tip_2 = l_tmp_tips[pos_tp_2]

            # Sélectionner celui qui a la distance la plus longue.
            # en faire le chemin 
            if tip_1!=-1 and tip_2!=-1:
                first_path = nx.shortest_path(G2, source=tip_1, target=tip_2)
                nrn_activated = self.get_nrn_from_path(first_path)

                # Supprime la liste des neurones:
                for nrn_id in nrn_activated:
                    try:
                        G2.remove_node(nrn_id)
                    except:
                        pass
                        # print("The node",nrn_id,"is not in the graph.")
                # Récupère la liste des neurones mobilisés et fait la différence ''
                # print("cp_lst_nrns\n", cp_lst_nrns)
                cp_lst_nrns = list(set(cp_lst_nrns).difference(nrn_activated))
                l_tmp_tips = cp_lst_nrns

                nb = self.nrn_tls.add_new_nrn("sentive_sequence_nrn")
                nrn3 = self.nrn_tls.lst_nrns[nb].neuron
                nrn3["meta"]["path"] = first_path
                nrn3["meta"]["mobilise_nrn2_ids"] = nrn_activated
                nrn3["DbConnectivity"]["pre_synaptique"] = nrn_activated
                nrn3["meta"]["ratio_pxls_total"] = len(nrn_activated)/self.nb_nrn_pxls
                self.nrn_segments.append(nrn3)
                # print("nouveau neurone 3 créé:", nb,", path:", first_path)
                print("taille neurones à séquencer :", len(cp_lst_nrns))

                if nb_min ==-1:
                    nb_min = nb
                reste_percent = 100*len(cp_lst_nrns)/nb_max
            else:
                print("cannot create any layer_3 neuron", tmp_max_length)

            int_limit -= 1

        # print
        segmented_path = self.nrn_tls.lst_nrns[nb_min].neuron["meta"]["path"]
        
        for nrn_pos in range(nb_min+1, len(self.nrn_tls.lst_nrns)):
            crnt_nrn = self.nrn_tls.lst_nrns[nrn_pos].neuron
            path_crnt = crnt_nrn["meta"]["path"]
            path_f1rst = self.nrn_tls.lst_nrns[nb_min].neuron["meta"]["mobilise_nrn2_ids"]
            # print("path_f1rst",path_f1rst)
            tip_1 = path_crnt[0]
            # récupérer les neurones connectés au tip_1
            candidates = self.get_nrn_from_path([tip_1])
            lst_candidats = set(path_f1rst).intersection(candidates)
            max_conno = 0
            nrn_win_1 = -1
            # print("lst_candidats",lst_candidats)
            for nrn2_id in lst_candidats:
                nrn2 = self.nrn_tls.get_neuron_from_id(nrn2_id)
                if max_conno < len(nrn2["DbConnectivity"]["lateral_connexion"]):
                    max_conno = len(nrn2["DbConnectivity"]["lateral_connexion"])
                    nrn_win_1 = nrn2_id
            
            # print("nrn_win 1",nrn_win_1)
            if nrn_win_1!=-1:
                crnt_nrn["meta"]["path"] = [nrn_win_1] + crnt_nrn["meta"]["path"]
                # boucle sur le path du first
                int_pos = 0
                for nrn_id in segmented_path:
                    # regarde les connexions de chaque neurone
                    nrn_connected = self.get_nrn_from_path([nrn_id])
                    if len(nrn_connected.intersection({nrn_win_1}))>0:
                        pos_2_cut_off["nrn_id"].append(nrn_id)
                        pos_2_cut_off["connector"].append(nrn_win_1)
                        pos_2_cut_off["position"].append(int_pos)
                        pos_2_cut_off["crnt_nrn"].append(crnt_nrn)
                        # bl_got_segmented = True
                        print("found possible connexion on nrn id",nrn_id, ":",nrn_connected)
                        break
                    int_pos += 1
            
            tip_2 = path_crnt[len(path_crnt)-1]
            candidates = self.get_nrn_from_path([tip_2])
            lst_candidats = set(path_f1rst).intersection(candidates)
            # print("lst_candidats",lst_candidats)
            max_conno = 0
            nrn_win_2 = -1
            
            for nrn2_id in lst_candidats:
                nrn2 = self.nrn_tls.get_neuron_from_id(nrn2_id)
                if max_conno < len(nrn2["DbConnectivity"]["lateral_connexion"]):
                    max_conno = len(nrn2["DbConnectivity"]["lateral_connexion"])
                    nrn_win_2 = nrn2_id

            if nrn_win_2!=-1 and nrn_win_2!=nrn_win_1:
                crnt_nrn["meta"]["path"].append(nrn_win_2)
                int_pos = 0
                for nrn_id in segmented_path:
                    # regarde les connexions de chaque neurone
                    nrn_connected = self.get_nrn_from_path([nrn_id])
                    if len(nrn_connected.intersection({nrn_win_2}))>0:
                        pos_2_cut_off["nrn_id"].append(nrn_id)
                        pos_2_cut_off["connector"].append(nrn_win_2)
                        pos_2_cut_off["position"].append(int_pos)
                        pos_2_cut_off["crnt_nrn"].append(crnt_nrn)
                        # bl_got_segmented = True
                        print("found possible connexion on nrn id",nrn_id, ":",nrn_connected)
                        break
                    int_pos += 1
            
        # Découper le segment principal en commençant par la position la plus petite
        print("Recherche des nœuds suivants", pos_2_cut_off["connector"], pos_2_cut_off["nrn_id"])
        # self.nrn_tls.remove_nrn_by_id(common_nrn)
        print("Commence la découpe de segmented_path",segmented_path)
        while len(pos_2_cut_off["position"])>0:
            int_pos = np.argmin(pos_2_cut_off["position"])
            shorter_path = []
            # print("recherche:",pos_2_cut_off["nrn_id"][int_pos])
            len_path = np.where(np.array(segmented_path) == pos_2_cut_off["nrn_id"][int_pos])
            if np.shape(len_path[0])[0]>0 :
                len_path = len_path[0][0]
            else:
                pos_2_cut_off["nrn_id"].pop(int_pos)
                pos_2_cut_off["connector"].pop(int_pos)
                pos_2_cut_off["position"].pop(int_pos)
                pos_2_cut_off["crnt_nrn"].pop(int_pos)
                continue
            for path_pos in range(len_path+1):
                new_path_nrn_id = segmented_path.pop(0)
                shorter_path.append(new_path_nrn_id)
                if new_path_nrn_id==pos_2_cut_off["nrn_id"][int_pos]:
                    shorter_path.append(pos_2_cut_off["connector"][int_pos])
                    segmented_path = [pos_2_cut_off["connector"][int_pos]] + segmented_path
                    # print("recherche",pos_2_cut_off["nrn_id"][int_pos],"segmented_path",segmented_path)
                    # Create le nrn3
                    nb = self.nrn_tls.add_new_nrn("sentive_sequence_nrn")
                    nrn3 = self.nrn_tls.lst_nrns[nb].neuron
                    nrn3["meta"]["path"] = shorter_path
                    nrn_activated = self.get_nrn_from_path(shorter_path)
                    nrn3["DbConnectivity"]["pre_synaptique"] = nrn_activated
                    nrn3["meta"]["ratio_pxls_total"] = len(nrn_activated)/self.nb_nrn_pxls
                    print("ratio_pxls_total nrn _ID",nrn3["_id"],", ",nrn3["meta"]["ratio_pxls_total"])
                    self.nrn_segments.append(nrn3)
                    bl_got_segmented = True
                    break
            pos_2_cut_off["nrn_id"].pop(int_pos)
            pos_2_cut_off["connector"].pop(int_pos)
            pos_2_cut_off["position"].pop(int_pos)
            pos_2_cut_off["crnt_nrn"].pop(int_pos)

        if bl_got_segmented:
            print("Fin de la découpe de segmented_path",segmented_path)
            self.nrn_segments.pop(0)
            self.nrn_tls.remove_nrn_pos(nb_min)
            nb = self.nrn_tls.add_new_nrn("sentive_sequence_nrn")
            nrn3 = self.nrn_tls.lst_nrns[nb].neuron
            nrn3["meta"]["path"] = segmented_path
            nrn_activated = self.get_nrn_from_path(segmented_path)
            nrn3["DbConnectivity"]["pre_synaptique"] = nrn_activated
            nrn3["meta"]["ratio_pxls_total"] = len(nrn_activated)/self.nb_nrn_pxls
            print("ratio_pxls_total nrn _ID",nrn3["_id"],", ",nrn3["meta"]["ratio_pxls_total"])
            self.nrn_segments.append(nrn3)

        print("nombre de neurones couches 1, 2 et 3 :",len(self.nrn_tls.lst_nrns))
        print("*"*40)



    def layer_4(self):
        """
            Création d'un ensemble de neurones permettant de comparer ensuite les caractères
        """
        # création d'une nouvelle couche 
        self.nrn_tls.new_layer()

        len_nrn = len(self.nrn_tls.lst_nrns)
        for nrn3_pos in range(self.nrn_tls.pos_nrn_by_layer[2], len_nrn):
            nrn3 = self.nrn_tls.lst_nrns[nrn3_pos].neuron
            # check 
            if nrn3["layer_id"]==3:
                for nrn2_pos in range(1, len(nrn3["meta"]["path"])-1):
                    nrn2_id = nrn3["meta"]["path"][nrn2_pos]
                    # crée un nouveau neurone
                    self.new_angle_neuron(nrn3["meta"]["path"][nrn2_pos-1], nrn2_id, nrn3["meta"]["path"][nrn2_pos+1])

                for nrn3_pos2 in range(nrn3_pos+1, len_nrn):
                    nrn3_2 = self.nrn_tls.lst_nrns[nrn3_pos2].neuron
                    # print(nrn3_2)
                    # Vérifie maintenant les 4 extrémités
                    if nrn3["meta"]["path"][0] == nrn3_2["meta"]["path"][0]:
                        # Crée un neurone commun
                        self.new_angle_neuron(nrn3["meta"]["path"][1], nrn3["meta"]["path"][0], nrn3_2["meta"]["path"][1])
                    elif nrn3["meta"]["path"][0] == nrn3_2["meta"]["path"][-1]:
                        self.new_angle_neuron(nrn3["meta"]["path"][1], nrn3["meta"]["path"][0], nrn3_2["meta"]["path"][-2])

                    elif nrn3["meta"]["path"][-1] == nrn3_2["meta"]["path"][-1]:
                        self.new_angle_neuron(nrn3["meta"]["path"][-2], nrn3["meta"]["path"][-1], nrn3_2["meta"]["path"][-2])
                    elif nrn3["meta"]["path"][-1] == nrn3_2["meta"]["path"][0]:
                        self.new_angle_neuron(nrn3["meta"]["path"][-2], nrn3["meta"]["path"][-1], nrn3_2["meta"]["path"][1])
        
        # Création des connexions latérales
        len_nrn = len(self.nrn_tls.lst_nrns)
        for nrn4_pos in range(self.nrn_tls.pos_nrn_by_layer[3], len_nrn):
            nrn4 = self.nrn_tls.lst_nrns[nrn4_pos].neuron
            # check 
            if nrn4["layer_id"]==4:
                for nrn2_id in nrn4["DbConnectivity"]["pre_synaptique"]:
                    nrn2 = self.nrn_tls.get_neuron_from_id(nrn2_id)
                    for nrn_lateral_id in nrn2["DbConnectivity"]["post_synaptique"]:
                        self.nrn_tls.add_nrn_lateral(nrn4, nrn_lateral_id)


    def new_angle_neuron(self, nrn2_1_id, nrn2_2_id, nrn2_3_id):
        # crée un nouveau neurone
        nb = self.nrn_tls.add_new_nrn("sentive_angle_neuron")
        nrn4 = self.nrn_tls.lst_nrns[nb].neuron
        nrn2_1 = self.nrn_tls.get_neuron_from_id(nrn2_1_id)
        nrn2_2 = self.nrn_tls.get_neuron_from_id(nrn2_2_id)
        nrn2_3 = self.nrn_tls.get_neuron_from_id(nrn2_3_id)
        self.nrn_tls.add_nrn_connexion(nrn4, nrn2_1)
        self.nrn_tls.add_nrn_connexion(nrn4, nrn2_2)
        self.nrn_tls.add_nrn_connexion(nrn4, nrn2_3)
        nrn4["meta"]["orientation"]["x"] = nrn2_3["meta"]["center"]["x"]-nrn2_1["meta"]["center"]["x"]
        nrn4["meta"]["orientation"]["y"] = nrn2_3["meta"]["center"]["y"]-nrn2_1["meta"]["center"]["y"]
        v1 = {"x":0,"y":0}
        v1["x"] = nrn2_2["meta"]["center"]["x"]-nrn2_1["meta"]["center"]["x"]
        v1["y"] = nrn2_2["meta"]["center"]["y"]-nrn2_1["meta"]["center"]["y"]
        v2 = {"x":0,"y":0}
        v2["x"] = nrn2_3["meta"]["center"]["x"]-nrn2_2["meta"]["center"]["x"]
        v2["y"] = nrn2_3["meta"]["center"]["y"]-nrn2_2["meta"]["center"]["y"]
        nrn4["meta"]["angle"] = self.nrn_tls.calc_angle(v1,v2)
        nrn4["meta"]["before_length"] = self.nrn_tls.calc_dist(nrn2_1["meta"]["center"], nrn2_2["meta"]["center"])
        nrn4["meta"]["after_length"] = self.nrn_tls.calc_dist(nrn2_3["meta"]["center"], nrn2_2["meta"]["center"])

    

    def run_layers(self):
        self.layer_1() # pixels
        self.layer_2() # triplets
        self.layer_3() # séquences, segments
        self.layer_4() # binomes -> caractères


    def reset_episode(self):
        self.episode[:,:,0]=self.episode[:,:,1]


    def show_neuron_receptive_field(self, nrn_id, verbose=False):

        rcptv_fields = self.nrn_tls.get_neuron_receptive_field(nrn_id, self.episode[:,:,0], self.nrn_tls.lst_nrns, verbose)
        
        plt.matshow(rcptv_fields)
        self.reset_episode()


    def show_receptive_field(self, neuron_idx,):
        # Visualiser le champs récepteur du neurone
        current_neuron = self.nrn_tls.lst_nrns[neuron_idx].neuron
        sub_matrix = self.nrn_tls.get_neuron_receptive_field(current_neuron, self.episode)
        print(current_neuron)
        plt.matshow(sub_matrix)

    
    def show_all_fields(self,lint_width=-1):
        if lint_width ==-1:
            all_fields = self.nrn_tls.get_all_center_fields(self.lst_nrns, self.episode)
        else:
            all_fields = self.nrn_tls.get_all_center_fields_width(self.lst_nrns, self.episode,lint_width)
        # print(all_fields)
        plt.matshow(all_fields)
        self.reset_episode()


    def show_receptive_field_id(self, neuron_idx2):
        # Visualiser le champs récepteur du neurone

        sub_matrix = self.nrn_tls.get_neuron_receptive_field(neuron_idx2, self.episode[:,:,0])
        # print(current_neuron)
        plt.matshow(sub_matrix)


    def print_neurons_by_layer_id(self, layer_id):
        for item in self.nrn_tls.lst_nrns:
            if item.neuron["layer_id"]==layer_id:
                print(item.neuron["_id"],":",item.neuron["DbConnectivity"]["pre_synaptique"], item.neuron)

    
    def draw_path_from_nrns_id(self, path, ax=-1):
        if ax==-1:
            _, ax = plt.subplots()
        x_values =[]
        y_values = []
        for nrn2_id in path:
            nrn2 = self.nrn_tls.get_neuron_from_id(nrn2_id)
            if nrn2=='': continue
            x_values.append(nrn2["meta"]["center"]["x"])
            y_values.append(nrn2["meta"]["center"]["y"])
        ax.plot(x_values, y_values, "k+-")


    def show_nrn_path_by_id(self, nrn_id, ax=-1):
        nrn = self.nrn_tls.get_neuron_from_id(nrn_id)
        return self.draw_path_from_nrns_id(nrn["meta"]["path"], ax)


    def draw_binome_angle_by_id(self, pos_id, ax=-1):
        if ax==-1:
            _, ax = plt.subplots()
        ax.plot(self.nrn_saccade[pos_id]["angles"],"kx-")
        ax.plot(self.nrn_saccade[pos_id]["l_angles"],"r*--")
        # ax.plot(np.abs(self.nrn_saccade[pos_id]["joints"]),"g+--")
        # print("ecart-type",np.std(self.nrn_saccade[pos_id]["angles"]))
        print("ratio_pxls_total",self.nrn_saccade[pos_id]["ratio_pxls_total"])
        ax.grid(True, which='both')

        ax.axhline(y=0, color='k')
        ax.axvline(x=0, color='k')


    def draw_segment_path_by_pos(self, nrn_pos, ax=-1):
        return self.draw_path_from_nrns_id(self.nrn_segments[nrn_pos]["meta"]["path"], ax)

    
    def draw_binome_path_by_pos(self, nrn_pos, ax=-1):
        return self.draw_path_from_nrns_id(self.nrn_saccade[nrn_pos]["path"], ax)

    
    def draw_selected_segment_path(self, ax=-1):
        if ax==-1:
            _, ax = plt.subplots()
        for nrn_id in self.slct_sgmts:
            nrn = self.nrn_tls.get_segment_from_id(nrn_id,self.nrn_segments)
            if nrn != '':
                self.draw_path_from_nrns_id(nrn["meta"]["path"], ax)


    def show_selected_segment_pxl(self, pos=-1, ax=-1):
        if ax==-1:
            _, ax = plt.subplots()
        np_stamp = np.zeros([self.IMG_SIZE,self.IMG_SIZE])
        i = 0
        for nrn_id in self.slct_sgmts:
            # for nrn_pxl_id in nrn["nrn3"]["meta"]["mobilise_pxl_ids"]:
            if pos==-1 or pos==i:
                nrn = self.nrn_tls.get_segment_from_id(nrn_id,self.nrn_segments)
                # print (i, nrn)
                for nrn_pxl_id in nrn["nrn3"]["meta"]["mobilise_nrn2_ids"]:
                    nrnx = self.nrn_tls.get_neuron_from_id(nrn_pxl_id)
                    
                    if nrnx != '': 
                        np_stamp[nrnx["meta"]["center"]["y"],nrnx["meta"]["center"]["x"]]+=1
                for nrn_pth_id in nrn["meta"]["path"]:
                    nrnx = self.nrn_tls.get_neuron_from_id(nrn_pth_id)
                    if nrnx != '': 
                        np_stamp[nrnx["meta"]["center"]["y"],nrnx["meta"]["center"]["x"]]+=1
            if pos!=-1 and pos==i:
                break
            i+=1
        ax.matshow(np_stamp)
