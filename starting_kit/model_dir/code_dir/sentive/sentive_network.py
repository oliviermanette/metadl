import copy

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

import networkx as nx

from .sentive_neuron_helper import sentive_neuron_helper

class sentive_network():

    def __init__(self, episode, epis_id=0):

        self.episd_id = epis_id

        self.episode = episode
        plt.matshow(episode[0][epis_id][:,:,0])
        
        ###########################################
        # meta parameters
        self.SEUIL = 0.5

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

        self.MIN_PATH = 10

        # end metaparameters
        ###########################################

        # nb est le premier identifiant pour créer les neurones
        self.nb = 0
        # liste contenant tous les neurones : pool_vision
        # self.pool_vision = []
        self.nrnl_map = np.zeros([self.IMG_SIZE,self.IMG_SIZE])
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


    def layer_1(self):
        ##################################################
        ######## NEURONES DE LA COUCHE 1 (t_1) #########
        ##################################################
        self.nrn_tls.new_layer()
        # Crée un neurone par pixel au début:
        pxl_coord = []
        for y in range(1,self.IMG_SIZE-1):
            for x in range(1,self.IMG_SIZE-1):
                if self.episode[0][self.episd_id][y][x][0]>self.SEUIL:
                    nb  = self.nrn_tls.add_new_nrn()
                    
                    self.nrn_tls.lst_nrns[nb].neuron["meta"]["center"]["x"] = x
                    self.nrn_tls.lst_nrns[nb].neuron["meta"]["center"]["y"] = y
                    self.nrn_tls.lst_nrns[nb].neuron["meta"]["matrix_width"] = 1
                    
                    self.nrnl_map[y][x] = nb

                    pxl_coord.append([x,y])
        print("nombre de neurones taille 1:",self.nrn_tls.nb_nrns)
        # print("*"*40,"\n")

        pca = PCA(n_components=2)
        pca.fit(pxl_coord)
        # on obtient les résultats ici:
        print(pca.components_)
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

        for neuron_idx in range(self.nrn_tls.nb_nrns):
            # position du centre du neurone
            x = self.nrn_tls.lst_nrns[neuron_idx].neuron["meta"]["center"]["x"]
            y = self.nrn_tls.lst_nrns[neuron_idx].neuron["meta"]["center"]["y"]
            sub_matrix = self.episode[0][self.episd_id][y-1:y+2, x-1:x+2, 0]
            sub_matrix2 = self.nrnl_map[y-1:y+2, x-1:x+2]
            
            int_nb_conf = 0
            sum_angl = 0
            vector_1 = {"x":0,"y":0}
            vector_2 = {"x":0,"y":0}
            local_tip_1 = {"x":0,"y":0}
            local_tip_2 = {"x":0,"y":0}
            main_vector = {"x":0,"y":0}
            list_presyn = []
            for ok_idx in range(len(self.nrn_tls.ok_conf)):
                # s'il n'y a pas de pixels en dehors de l'axe de symétrie, y a rien à calculer.
                trois_mtrx = np.multiply(sub_matrix>self.SEUIL, self.nrn_tls.ok_conf[ok_idx])
                # trois_mtrx = np.multiply(sub_matrix2, self.nrn_tls.ok_conf[ok_idx])
                if np.sum(np.abs(trois_mtrx))==3:
                    
                    ###########################
                    # Détermine les extrémités
                    # retire le pixel central
                    trois_mtrx = np.multiply(sub_matrix2, self.nrn_tls.ok_conf[ok_idx])
                    
                    list_presyn.extend(np.absolute(trois_mtrx).ravel())

                    trois_mtrx = np.multiply(sub_matrix, self.nrn_tls.ok_conf[ok_idx])
                    sub_result = np.multiply(trois_mtrx, self.nrn_tls.dir_matrix)
                    
                    #######
                    # d'un côté de l'axe (valeurs positives)
                    tmp_coord = np.trim_zeros(np.multiply(sub_result>0,self.nrn_tls.get_y_matrix(3)).ravel())
                    if len(tmp_coord)>0:
                        local_tip_1["y"] = np.mean(tmp_coord)
                    tmp_coord = np.trim_zeros(np.multiply(sub_result>0,self.nrn_tls.get_x_matrix(3)).ravel())
                    if len(tmp_coord)>0:
                        local_tip_1["x"] = np.mean(tmp_coord)
                    ##
                    # calcul des vecteurs directeurs
                    vector_1["y"] = self.nrn_tls.get_matrix_center(3) - local_tip_1["y"]
                    vector_1["x"] = self.nrn_tls.get_matrix_center(3) - local_tip_1["x"]
                    
                    #######
                    # les pixels de l'autre côté de l'axe les valeurs sont négatives
                    tmp_coord = np.trim_zeros(np.multiply(sub_result<0,self.nrn_tls.get_y_matrix(3)).ravel())
                    if len(tmp_coord)>0: 
                        local_tip_2["y"] = np.mean(tmp_coord)
                    tmp_coord = np.trim_zeros(np.multiply(sub_result<0,self.nrn_tls.get_x_matrix(3)).ravel())
                    if len(tmp_coord)>0:
                        local_tip_2["x"] = np.mean(tmp_coord)
                    ##
                    # calcul des vecteurs directeurs 
                    vector_2["y"] = local_tip_2["y"] - self.nrn_tls.get_matrix_center(3)
                    vector_2["x"] = local_tip_2["x"] - self.nrn_tls.get_matrix_center(3)
                    
                    #######
                    tmp_angl = self.nrn_tls.calc_angle(vector_1, vector_2)

                    if not np.isnan(tmp_angl):
                        main_vector["x"] += vector_2["x"] + vector_1["x"]
                        main_vector["y"] += vector_2["y"] + vector_1["y"]
                        sum_angl += tmp_angl
                        int_nb_conf += 1

                    #######
            #####################################
            
            if int_nb_conf>0:
                # crée un nouveau neurone de taille 3
                nb  = self.nrn_tls.add_new_nrn()
                lst_nrn2_pos.append(nb)
                self.nrn_tls.lst_nrns[nb].neuron["meta"]["center"]["x"] = x
                self.nrn_tls.lst_nrns[nb].neuron["meta"]["center"]["y"] = y
                self.nrn_tls.lst_nrns[nb].neuron["meta"]["matrix_width"] = 3
                self.nrn_tls.lst_nrns[nb].neuron["DbConnectivity"]["pre_synaptique"].extend(set(np.array(list_presyn).astype(int)))

                for i in range(len(self.nrn_tls.lst_nrns[nb].neuron["DbConnectivity"]["pre_synaptique"])-1,-1,-1):
                    if self.nrn_tls.lst_nrns[nb].neuron["DbConnectivity"]["pre_synaptique"][i]==0:
                        self.nrn_tls.lst_nrns[nb].neuron["DbConnectivity"]["pre_synaptique"].pop(i)
                    else:
                        self.nrn_tls.netGraph.add_edge(self.nrn_tls.lst_nrns[nb].neuron["_id"],self.nrn_tls.lst_nrns[nb].neuron["DbConnectivity"]["pre_synaptique"][i])
                        # self.nrn_tls.add_edge()
                        nrn_pxl = self.nrn_tls.get_neuron_from_id(self.nrn_tls.lst_nrns[nb].neuron["DbConnectivity"]["pre_synaptique"][i])
                        nrn_pxl["DbConnectivity"]["post_synaptique"].append(self.nrn_tls.lst_nrns[nb].neuron["_id"])
                if int_nb_conf>0: 
                    self.nrn_tls.lst_nrns[nb].neuron["meta"]["angle"] = sum_angl/int_nb_conf
                    self.nrn_tls.lst_nrns[nb].neuron["meta"]["vector_1"]["x"] =  main_vector["x"]/int_nb_conf
                    self.nrn_tls.lst_nrns[nb].neuron["meta"]["vector_1"]["y"] =  main_vector["y"]/int_nb_conf

        # détermine les connexions latérales des neurones de la couche 2
        
        for nrn2_pos in lst_nrn2_pos:
            str_next_nrns = {}
            # print("\nneurone ID:", self.nrn_tls.lst_nrns[nrn2_pos].neuron["_id"])
            for nrn1_id in self.nrn_tls.lst_nrns[nrn2_pos].neuron["DbConnectivity"]["pre_synaptique"]:
                nrn2_not_id = self.nrn_tls.lst_nrns[nrn2_pos].neuron["_id"]
                nrn1 = self.nrn_tls.get_neuron_from_id(nrn1_id)
                for nrn2_id in nrn1["DbConnectivity"]["post_synaptique"]:
                    if nrn2_id != nrn2_not_id:
                        self.nrn_tls.increment_weight(self.nrn_tls.lst_nrns[nrn2_pos].neuron, nrn2_id)
                        try:
                            str_next_nrns[nrn2_id] += 1
                        except KeyError:
                            str_next_nrns[nrn2_id] = 1
            # slct_max = max(str_next_nrns.values())
            next_keys = list(str_next_nrns.keys())
            # next_vals = list(str_next_nrns.values())
            lst_next_nrns = []
            for pos in range(len(next_keys)):
                # if slct_max == next_vals[pos]:
                    tmp_id = next_keys[pos]
                    lst_next_nrns.append(tmp_id)
                    self.nrn_tls.lst_nrns[nrn2_pos].neuron["DbConnectivity"]["post_synaptique"].append(tmp_id)
                    # self.nrn_tls.netGraph.add_edge(nrn2_not_id,tmp_id)
                    
                    self.nrn_tls.add_edge(nrn2_not_id,tmp_id)
        
        self.nrn_tls.nb_2_1st_layers = len(self.nrn_tls.lst_nrns)
        print("\nnombre de neurones couche 1 & 2, tailles 1 & 3:",self.nrn_tls.nb_2_1st_layers)
        # print("*"*40)



    def add_similar_nrn(self, nrn3, lst_nrn_ltrl, cp_lst_nrns, nrn_stop_id):
        for nrn2_id in lst_nrn_ltrl:
            if (nrn2_id == nrn_stop_id).any():
                nrn3["meta"]["nodes"].append(nrn2_id)
                return
            nrn2 = self.nrn_tls.get_neuron_from_id(nrn2_id, cp_lst_nrns)
            if nrn2 !='':
                nrn3["DbConnectivity"]["pre_synaptique"].append(nrn2_id)
                # print("nrn2_id",nrn2_id)
                self.nrn_tls.netGraph.add_edge(nrn3["_id"], nrn2_id)
                
                # récupère la liste des neurones latéraux pour les inspecter
                lst_nrn_ltrl = nrn2["DbConnectivity"]["post_synaptique"]
                self.nrn_tls.remove_nrn_by_id(nrn2_id, cp_lst_nrns)
                self.add_similar_nrn(nrn3, lst_nrn_ltrl, cp_lst_nrns, nrn_stop_id)

    
    def update_threshold(self, cp_lst_nrns):
        # *** Recherche à nouveau les TIPS et NœUDS dans les neurones restants
        nrn_ratio_conn = []
        nb_ltrl_conn = []
        for pos in range(len(cp_lst_nrns)):
            new_ratio = cp_lst_nrns[pos].neuron["ratio_conn"]
            nrn_ratio_conn.append(new_ratio)
            nb_ltrl_conn.append(len(cp_lst_nrns[pos].neuron["DbConnectivity"]["post_synaptique"]))
        
        # seuils nombre absolu de connexion
        lthrshld_tip = min(nb_ltrl_conn) # seuil pour détecter le extrémités
        lthrshld_nod = max(nb_ltrl_conn) # seuil pour détecter le extrémités
        # seuil nombre relatif de connexions
        r_thrshld_tip = min(nrn_ratio_conn)
        return lthrshld_tip, lthrshld_nod, r_thrshld_tip


    def find_tips(self, cp_lst_nrns, lthrshld_tip, lthrshld_nod, G, r_thrshld_tip=-1):
        l_tmp_tips = [] # id des neurones situés à une extrémité
        l_tmp_node = [] # id des neurones au carrefour
        for pos in range(len(cp_lst_nrns)):
            nrn = cp_lst_nrns[pos].neuron
            # Sélection des neurones TIPS
            if len(nrn["DbConnectivity"]["post_synaptique"])<=lthrshld_tip:
                l_tmp_tips.append(nrn["_id"])
            if nrn["ratio_conn"]<=r_thrshld_tip:
                l_tmp_tips.append(nrn["_id"])
            # Sélection des neurones NODES
            if len(nrn["DbConnectivity"]["post_synaptique"])>=lthrshld_nod:
                l_tmp_node.append(nrn["_id"])
        
        l_tmp_tips = list(set(l_tmp_tips))
        longuest = []
        for t in l_tmp_tips:
            tmp_path_length = []
            for n in l_tmp_node:
                tmp_path_length.append(int(nx.shortest_path_length(G,t,n)))
            longuest.append(min(tmp_path_length))
        # réarrange les extrémités en fonction de la longueur des chemins les plus courts
        l_tmp_tips = np.array(l_tmp_tips)
        l_tmp_tips = l_tmp_tips[np.flip(np.argpartition(np.array(longuest),len(longuest)-1))]
        return l_tmp_tips, np.array(l_tmp_node)


    def new_sequence_neuron(self, cp_lst_nrns, l_tmp_tips, nrn_stop_id):
        # Crée un neurone séquence par l_tmp_tips
        for nrn2_id in l_tmp_tips:
            nb = self.nrn_tls.add_new_nrn("sentive_sequence_nrn")
            nrn3 = self.nrn_tls.lst_nrns[nb].neuron
            nrn3["DbConnectivity"]["pre_synaptique"].append(nrn2_id)
            nrn3["meta"]["tips"].append(nrn2_id)
            nrn2 = self.nrn_tls.get_neuron_from_id(nrn2_id, cp_lst_nrns)
            try:
                lst_nrn_ltrl = nrn2["DbConnectivity"]["post_synaptique"]
            except:
                print("*********",nrn2_id,"not found")
            self.nrn_tls.remove_nrn_by_id(nrn2_id, cp_lst_nrns)
            self.add_similar_nrn(nrn3, lst_nrn_ltrl, cp_lst_nrns, nrn_stop_id)
        # ################
        # supprime les neurones sequences mal-nés.
        nrn_to_pop = []
        for nrn3_pos in range(len(self.nrn_tls.lst_nrns)-1,self.nrn_tls.nb_2_1st_layers, -1):
            nrn3 = self.nrn_tls.lst_nrns[nrn3_pos].neuron
            if nrn3["layer_id"]==3:
                if len(nrn3["DbConnectivity"]["pre_synaptique"])<2:
                    lbl_found = False
                    for nrn2_id in nrn3["DbConnectivity"]["pre_synaptique"]:
                        nrn2 = self.nrn_tls.get_neuron_from_id(nrn2_id)
                        try:
                            if (np.array(nrn2["DbConnectivity"]["post_synaptique"])==np.array(nrn3["meta"]["nodes"])).any():
                                lbl_found = True
                                continue
                        except:
                            if (np.array(nrn2["DbConnectivity"]["post_synaptique"])==np.array(nrn3["meta"]["nodes"])):
                                lbl_found = True
                                continue
                    if not lbl_found:
                        nrn_to_pop.append(nrn3["_id"])
                        self.nrn_tls.remove_nrn_pos(nrn3_pos)


    def mobilise_nrn_path(self, nrn3):
        # pour chaque élement du path, il faut sélectionner le neurone et compiler les neurones post_synaptique
        path_nrn_lnk = []
        for nrn2_id in nrn3["meta"]["path"]:
            nrn2 = self.nrn_tls.get_neuron_from_id(nrn2_id)
            if nrn2 == '':
                print("Neuron not found",nrn2_id)
            else:
                path_nrn_lnk.extend(nrn2["DbConnectivity"]["post_synaptique"])
        path_nrn_lnk = set(path_nrn_lnk)
        set_nrn_seq = set(nrn3["DbConnectivity"]["pre_synaptique"])
        # print("neurone SEQUENCE #",nrn3["_id"],"-> nb total nrn mobilisés", len(set_nrn_seq),"sequence longueur:",len(nrn3["meta"]["path"]))
        dif_seq = set_nrn_seq.difference(path_nrn_lnk)
        reste_percent = 100*len(dif_seq)/len(set_nrn_seq)
        # print("après shortest path", len(dif_seq),"soit",reste_percent,"% restant")

        common_nrn = {}
        if reste_percent>self.MIN_PATH:
            # recherche le dernier neurone en contact avec la branche alternative
            lst_nrn_ltrl = []
            for nrn2_id in dif_seq:
                nrn2 = self.nrn_tls.get_neuron_from_id(nrn2_id)
                lst_nrn_ltrl.extend(nrn2["DbConnectivity"]["post_synaptique"])
            lst_nrn_ltrl = set(lst_nrn_ltrl)
            common_nrn = set(path_nrn_lnk).intersection(lst_nrn_ltrl)
            # print("common_nrn",common_nrn)

            lst_nrn_ltrl = []
            for nrn2_id in common_nrn:
                nrn2 = self.nrn_tls.get_neuron_from_id(nrn2_id)
                if nrn2=='':
                    print("Neuron not found",nrn2_id)
                else:
                    lst_nrn_ltrl.extend(nrn2["DbConnectivity"]["post_synaptique"])
            lst_nrn_ltrl = set(lst_nrn_ltrl)
            common_nrn = set(nrn3["meta"]["path"]).intersection(lst_nrn_ltrl)
            # print("common_nrn2",common_nrn)

        return reste_percent, common_nrn


    def get_best_node(self, common_nrn):
        flt_best_ratio = 0
        int_best_nb = 0
        output_nrn_id = -1
        for nrn2_id in common_nrn:
            nrn2 = self.nrn_tls.get_neuron_from_id(nrn2_id)
            if len(nrn2["DbConnectivity"]["post_synaptique"]) == int_best_nb:
                if nrn2["ratio_conn"]>flt_best_ratio:
                    flt_best_ratio = nrn2["ratio_conn"]
                    output_nrn_id = nrn2_id 
            elif len(nrn2["DbConnectivity"]["post_synaptique"]) > int_best_nb:
                int_best_nb = len(nrn2["DbConnectivity"]["post_synaptique"])
                flt_best_ratio = nrn2["ratio_conn"]
                output_nrn_id = nrn2_id 
        return output_nrn_id

 

    def layer_3(self):
        """ Détermine * les neurones séquences *
        """
        # création d'une nouvelle couche 
        self.nrn_tls.new_layer()
        
        # lien vers la 2eme couche:
        G = self.nrn_tls.layer_graph[1]

        cp_lst_nrns = [] # copie de la liste des neurones de la 2eme couche
        # calcul des ratios pour chaque neurone
        nrn_ratio_conn = []
        nb_ltrl_conn = []
        for nrn_pos in self.nrn_tls.lst_nrns:
            nrn = nrn_pos.neuron
            if nrn["layer_id"]==2:
                cp_lst_nrns.append(copy.deepcopy(nrn_pos))
                nb_ltrl_conn.append(len(nrn["DbConnectivity"]["post_synaptique"]))
                tmp_num_conn = []                
                for nrn_lat in nrn["DbConnectivity"]["post_synaptique"]:
                    tmp_num_conn.append(len(self.nrn_tls.lst_nrns[nrn_lat-1].neuron["DbConnectivity"]["post_synaptique"]))
                new_ratio = len(nrn["DbConnectivity"]["post_synaptique"])/np.mean(tmp_num_conn)
                nrn["ratio_conn"] = new_ratio
                nrn_ratio_conn.append(new_ratio)

        # save_cp_lst_nrns = copy.deepcopy(cp_lst_nrns)
        
        # seuil nombre relatif de connexions
        r_thrshld_tip = min(nrn_ratio_conn)
        lthrshld_tip = min(nb_ltrl_conn) # seuil pour détecter le extrémités
        lthrshld_nod = max(nb_ltrl_conn) # seuil pour détecter le extrémités

        # print("taille neurones à séquencer :", len(cp_lst_nrns))

        l_tmp_tips, nrn_stop_id = self.find_tips(cp_lst_nrns, lthrshld_tip, lthrshld_nod, G)
        # print("neurons TIPS:",l_tmp_tips,"et neurons NODE:",nrn_stop_id)
        initial_tips = copy.deepcopy(l_tmp_tips)

        self.new_sequence_neuron(cp_lst_nrns, l_tmp_tips, nrn_stop_id)
        # ****************************************************************
        # print("neurones restants à séquencer", len(cp_lst_nrns))
        
        lthrshld_tip, lthrshld_nod, r_thrshld_tip = self.update_threshold(cp_lst_nrns)

        l_tmp_tips, nrn_stop_id = self.find_tips(cp_lst_nrns, lthrshld_tip, lthrshld_nod, G, r_thrshld_tip)
        # print("neurons TIPS:", l_tmp_tips,"et neurons NODE:", nrn_stop_id)

        self.new_sequence_neuron(cp_lst_nrns, l_tmp_tips, nrn_stop_id)
         # ****************************************************************
        # print("neurones restants à séquencer", len(cp_lst_nrns))

        for nrn3_pos in range(len(self.nrn_tls.lst_nrns)):
            try:
                nrn3 = self.nrn_tls.lst_nrns[nrn3_pos].neuron
            except IndexError:
                continue
            if nrn3["layer_id"]==3:
                nrn3_tips_lst = nrn3["meta"]["tips"]
                nrn3_nodes_lst = np.array(list(set(nrn3["meta"]["nodes"])))
                tmp_path_length = []
                for n in nrn3_nodes_lst:
                    tmp_path_length.append(int(nx.shortest_path_length(G,nrn3_tips_lst[0],n)))
                nrn3_nodes_lst = nrn3_nodes_lst[np.argpartition(np.array(tmp_path_length),len(tmp_path_length)-1)]
                clst_node = nrn3_nodes_lst[0]

                nrn3["meta"]["path"] = nx.shortest_path(G,nrn3_tips_lst[0],clst_node)
                reste_percent, common_nrn = self.mobilise_nrn_path(nrn3)
                pos_test = 0

                common_nrn = self.get_best_node(common_nrn)
                dest_lst = np.array([common_nrn])

                while reste_percent>self.MIN_PATH:
                    # Je tiens le coupable c'est common_nrn
                    
                    # supprime le nœud dans la liste générale et la liste locale
                    # print("remove nrn",common_nrn,self.nrn_tls.remove_nrn_by_id(common_nrn))
                    # cp_lst_nrns = copy.deepcopy(save_cp_lst_nrns)
                    self.nrn_tls.remove_nrn_by_id(common_nrn, cp_lst_nrns)
                    # tu prends comme nouvelle destination le prochain nœud de la liste... nrn3_nodes_lst
                    
                    dest_lst = np.append(dest_lst, nrn3["meta"]["tips"], 0)
                    # si le neurone coupable de ce raccourci est en fait le node il faut choisir une autre destination
                    if list(set(nrn3["meta"]["nodes"])) == [common_nrn]:
                        # On choisit la nouvelle destination dans la liste des nodes suivants:
                        if len(nrn3_nodes_lst)>1:
                            nrn3_nodes_lst.pop(0)
                            clst_node = nrn3_nodes_lst[0]
                            nrn3["meta"]["nodes"] = [clst_node]
                            nrn3["meta"]["path"] = nx.shortest_path(G,nrn3_tips_lst[0],clst_node)
                            reste_percent, common_nrn  = self.mobilise_nrn_path(nrn3)
                        else:
                            # sinon, sélectionne un nouveau tip dans initial_tips
                            while pos_test<= len(initial_tips) -1:
                                new_dest = initial_tips[pos_test]
                                # Il ne faut pas que cette destination ait déjà été testée
                                if (dest_lst == new_dest).any():
                                    pos_test += 1
                                else:
                                    # print("test4")
                                    np.append(dest_lst,new_dest)
                                    nrn3["meta"]["nodes"] = [new_dest]
                                    nrn3["meta"]["path"] = nx.shortest_path(G,nrn3_tips_lst[0],new_dest)
                                    reste_percent, common_nrn = self.mobilise_nrn_path(nrn3)
                                    # print("reste_percent, common_nrn",reste_percent, common_nrn)
                                    common_nrn = self.get_best_node(common_nrn)
                                    # print("common_nrn selectionné",common_nrn)
                                    break
                    else:
                        nrn3["meta"]["path"] = nx.shortest_path(G,nrn3_tips_lst[0],nrn3["meta"]["nodes"][0])
                        # print("new PATH",nrn3["meta"]["path"])
                        reste_percent, common_nrn = self.mobilise_nrn_path(nrn3)
                        # print("reste_percent, common_nrn", reste_percent, common_nrn)
                        if common_nrn != {}:
                            common_nrn = self.get_best_node(common_nrn)
                            # print("common_nrn", common_nrn)
                        else:
                            print("** fin (-: **")
                            break
                # Calcul de la longueur de chaque segment
                nrn3["meta"]["total_length"] = self.nrn_tls.calc_total_distance(nrn3["meta"]["path"])
                

        print("\nnombre de neurones couches 1, 2 et 3 :",len(self.nrn_tls.lst_nrns))
        print("*"*40)


    def layer_4(self):
        """Les neurones Segments 
        """
        contain_segment = []
        self.save_segments = contain_segment

        # création d'une nouvelle couche 
        self.nrn_tls.new_layer()
        # pour chaque neurone de la couche 3 recherche les segments possibles
        for nrn3_pos in range(len(self.nrn_tls.lst_nrns)):
            try:
                nrn3 = self.nrn_tls.lst_nrns[nrn3_pos].neuron
            except IndexError:
                continue
            if nrn3["layer_id"]==3:
                # récupère le path 
                contain_segment.append({
                    "path" : nrn3["meta"]["path"],
                    "vecteurs" : [],
                    "angles" : [],
                    "l_angles" : [], # angles lissés
                    "joints" : [] # position entre 2 segments
                    })
                tmp_pos = len(contain_segment)-1
                if len(contain_segment[tmp_pos]["path"])>1:
                    nrn2_id = contain_segment[tmp_pos]["path"][0]
                    nrn2 = self.nrn_tls.get_neuron_from_id(nrn2_id)
                    point1 = nrn2["meta"]["center"]
                    for nrn2_pos in range(1, len(contain_segment[tmp_pos]["path"])):
                        nrn2_id = contain_segment[tmp_pos]["path"][nrn2_pos]
                        nrn2 = self.nrn_tls.get_neuron_from_id(nrn2_id)
                        point2 = nrn2["meta"]["center"]
                        vecteur = {
                            "x":point2["x"]-point1["x"],
                            "y":point2["y"]-point1["y"]
                        }
                        contain_segment[tmp_pos]["vecteurs"].append(vecteur)
                        point1 = point2
                    
                    # calcul des angles
                    if len(contain_segment[tmp_pos]["vecteurs"])>1:
                        vector_1 = contain_segment[tmp_pos]["vecteurs"][0]
                        for v_pos in range(1, len(contain_segment[tmp_pos]["vecteurs"])):
                            vector_2 = contain_segment[tmp_pos]["vecteurs"][v_pos]
                            contain_segment[tmp_pos]["angles"].append(self.nrn_tls.calc_angle(vector_1, vector_2))
                            vector_1 = vector_2

                    # Calcul des angles lissés l_angles
                    window_half_length = 2
                    np_angles = contain_segment[tmp_pos]["angles"]
                    # print("np_angles taille:",np.shape(np_angles))
                    len_angles = len(contain_segment[tmp_pos]["angles"])
                    for angle_pos in range(len_angles):
                        start_pos = angle_pos - window_half_length
                        if start_pos < 0:
                            start_pos = 0
                        end_pos = angle_pos + window_half_length + 1
                        if end_pos>=len_angles:
                            end_pos = len_angles-1
                        # print("start",start_pos,"end",end_pos)
                        contain_segment[tmp_pos]["l_angles"].append(np.mean(np_angles[start_pos:end_pos]))

                    contain_segment[tmp_pos]["joints"] = np.zeros(np.shape(contain_segment[tmp_pos]["angles"]))
                    last_pos=1
                    lst_joints = []
                    if len(contain_segment[tmp_pos]["l_angles"])>0:
                        previous = contain_segment[tmp_pos]["l_angles"][0]
                        for angle_pos in range(1, len_angles):
                            if contain_segment[tmp_pos]["l_angles"][angle_pos]!=0:
                                first = previous/contain_segment[tmp_pos]["l_angles"][angle_pos]
                                previous = contain_segment[tmp_pos]["l_angles"][angle_pos]
                                if first <0:
                                    print("On vient de changer de signe là, on est à la position", angle_pos, contain_segment[tmp_pos]["l_angles"][angle_pos-1])
                                    if contain_segment[tmp_pos]["l_angles"][angle_pos-1]<0:
                                        # recherche dans les négatifs
                                        good_pos = np.argmin(contain_segment[tmp_pos]["angles"][last_pos:angle_pos])
                                        print("entre les positions",last_pos,"et", angle_pos, contain_segment[tmp_pos]["angles"][last_pos:angle_pos+1],"trouvé:",good_pos)
                                    else:
                                        good_pos = np.argmax(contain_segment[tmp_pos]["angles"][last_pos:angle_pos])
                                    contain_segment[tmp_pos]["joints"][good_pos+last_pos]=1
                                    lst_joints.append(good_pos+last_pos)
                                    last_pos = angle_pos
                    # Il faut maintenant créer les neurones segments 1+nb de 1 dans l_angles
                    nb_nrn_segments = 1 + np.sum(contain_segment[tmp_pos]["l_angles"])
                    nrn2 = self.nrn_tls.get_neuron_from_id(nrn3["meta"]["path"][0])
                    l_tip_1 = nrn2["meta"]["center"]
                    l_vctr_1 = contain_segment[tmp_pos]["vecteurs"][0]
                    # angle = 
                    # for i in range(nb_nrn_segments):
                    #     nb = self.nrn_tls.add_new_nrn()



        self.save_segments = contain_segment


                
        print("\nnombre de neurones couches 1, 2, 3 et 4:",len(self.nrn_tls.lst_nrns))
        print("*"*40)



    def run_layers(self):
        self.layer_1() # pixels
        self.layer_2() # triplets
        self.layer_3() # séquences
        self.layer_4() # segments
        # self.layer_5() # caractères


    def reset_episode(self, int_id=0):
        self.episode[0][int_id][:,:,0]=self.episode[0][int_id][:,:,1]


    def show_neuron_receptive_field(self, nrn_id, int_id=0, verbose=False):

        rcptv_fields = self.nrn_tls.get_neuron_receptive_field(nrn_id, self.episode[0][int_id][:,:,0], self.nrn_tls.lst_nrns, verbose)
        
        plt.matshow(rcptv_fields)
        self.reset_episode(int_id)


    def show_receptive_field(self, neuron_idx,int_id=0):
        # Visualiser le champs récepteur du neurone
        current_neuron = self.lst_nrns[neuron_idx].neuron
        sub_matrix = self.nrn_tls.get_receptive_field(current_neuron, self.episode[0][int_id])
        print(current_neuron)
        plt.matshow(sub_matrix)

    
    def show_all_fields(self, int_id=0, lint_width=-1):
        if lint_width ==-1:
            all_fields = self.get_all_center_fields(self.lst_nrns, self.episode[0][int_id][:,:,0])
        else:
            all_fields = self.get_all_center_fields_width(self.lst_nrns, self.episode[0][int_id][:,:,0],lint_width)
        # print(all_fields)
        plt.matshow(all_fields)
        self.reset_episode(int_id)


    def show_receptive_field_id(self, neuron_idx2,int_id=-1):
        # Visualiser le champs récepteur du neurone
        for neuron_idx in range(self.nrn_tls.nb_nrns):
            if self.nrn_tls.lst_nrns[neuron_idx].neuron["_id"]==neuron_idx2:
                break
        if int_id==-1: 
            int_id = self.episd_id
        current_neuron = self.nrn_tls.lst_nrns[neuron_idx].neuron
        sub_matrix = self.nrn_tls.get_receptive_field(current_neuron, self.episode[0][int_id])
        print(current_neuron)
        plt.matshow(sub_matrix)


    def show_neurons_by_layer_id(self, layer_id):
        for item in self.nrn_tls.lst_nrns:
            if item.neuron["layer_id"]==layer_id:
                print(item.neuron["_id"],":",item.neuron["DbConnectivity"]["pre_synaptique"], item.neuron)

    
    def show_sequence_by_id(self, nrn_id):
        nrn = self.nrn_tls.get_neuron_from_id(nrn_id)
        x_values =[]
        y_values = []
        for nrn2_id in nrn["meta"]["path"]:
            nrn2 = self.nrn_tls.get_neuron_from_id(nrn2_id)
            x_values.append(nrn2["meta"]["center"]["x"])
            y_values.append(nrn2["meta"]["center"]["y"])
        plt.plot(x_values, y_values, "k+-")
