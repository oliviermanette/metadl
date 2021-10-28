import numpy as np
import pandas as pd

from .sentive_network import sentive_network

class sentive_brain():
    def __init__(self, episode, nb_char):
        self.episode = episode
        self.nnet = []
        self.nb_char = nb_char
        for i in range(nb_char):
            print("\n********* network:",i)
            self.nnet.append(sentive_network(episode[0][i]))
            self.nnet[i].run_layers()
        
            # Dessine les segments détectés: 
            self.nnet[i].draw_selected_segment_path()

        # J'ai ici tous les binomes qui sont terminés.
        # Je dois les comparer avec les autres caractères.
        # Un boucle pour passer en revu chaque caractère
        for lnet in range(nb_char):
            # pour chaque caractère je fais une boucle sur chaque binomes.
            for bin1_id in range(len( self.nnet[lnet].nrn_saccade)): 
                # pour chaque binome je fais une boucle sur tous les autres caractères.
                    max_result = 100000000000
                    for o_net in range(nb_char):
                        # rien ne sert de comparer par rapport à lui-même
                        if o_net != lnet:
                            # Je fais le calcul pour chaque binome de l'autre caractères
                            for test_saccade_id in range(len( self.nnet[o_net].nrn_saccade)): 
                                # dans un sens ...
                                l_result, offset = self.nnet[lnet].nrn_tls.test_sequences(self.nnet[lnet].nrn_saccade[bin1_id],self.nnet[o_net].nrn_saccade[test_saccade_id])
                                if l_result < max_result:
                                    # if o_net==0:
                                    #     print("l_result",l_result)
                                    if l_result<0:
                                        print("l_result",l_result,"test_saccade_id",test_saccade_id,"character_id",lnet, "comparaison", o_net, "offset",offset)
                                    else:
                                        max_result = l_result
                                
                    # écrit le max_result dans le mini_score
                    self.nnet[lnet].nrn_saccade[bin1_id]["mini_score"] = max_result


    def predict(self, test_img):
        # print("I was here")
        self.test_net = sentive_network(test_img)
        self.test_net.run_layers()
        self.test_net.draw_selected_segment_path()
        results = []
        full_results = {}
        for lnet in range(self.nb_char):
            # pour chaque caractère je fais une boucle sur chaque binomes.
            best_result = []
            results_char = {}
            for memory_saccade_id in range(len( self.test_net.nrn_saccade)):
                min_error = []
                results_test = {}
                # pour chaque binome je fais une boucle sur tous les autres caractères.
                # Je fais le calcul pour chaque binome de l'autre caractères
                for test_saccade_id in range(len( self.nnet[lnet].nrn_saccade)): 
                    error, saved_offset = self.nnet[lnet].nrn_tls.test_sequences(self.nnet[lnet].nrn_saccade[test_saccade_id],self.test_net.nrn_saccade[memory_saccade_id])
                    error = error/self.nnet[lnet].nrn_saccade[test_saccade_id]["mini_score"]
                    results_test[test_saccade_id]=error
                    # error = self.nnet[lnet].nrn_tls.test_sequences(self.nnet[lnet].nrn_saccade[test_saccade_id],self.test_net.nrn_saccade[memory_saccade_id], True)/self.nnet[lnet].nrn_saccade[test_saccade_id]["mini_score"]
                    # error = self.nnet[lnet].nrn_tls.test_sequences(self.nnet[lnet].nrn_saccade[test_saccade_id],self.test_net.nrn_saccade[memory_saccade_id], True)
                    if not np.isnan(error):
                        min_error.append(error)
                    # print("test du binome ",memory_saccade_id,"avec le caractère", lnet, "et avec son binome ", test_saccade_id,"erreur:",error,"(",self.nnet[lnet].nrn_saccade[test_saccade_id]['ratio_pxls_total']*100,"%)")
                                             
                    # écrit le max_result dans le mini_score
                best_result.append(np.min(min_error))
                results_char[memory_saccade_id]=results_test
            results.append(np.mean(best_result))
            full_results[lnet]=pd.DataFrame(results_char)

        char_result = []
        for Id_char in range (len(full_results)):
            # Id_char = 2
            stored_min = []
            int_size = min(full_results[Id_char].shape)
            # print("char:", Id_char)
            if int_size==1:
                char_result.append(full_results[Id_char][0][0])
            else:
                for i_size in range (int_size, 1, -1):
                    # print("i_size",i_size)
                    new_tbl = full_results[Id_char].iloc[0:i_size,0:i_size]
                    local_store = []
                    while i_size>0:
                        # print(new_tbl)
                        local_store.append(new_tbl.min().min())
                        # suppression de la ligne et de la colonne du minimum
                        if i_size>1:
                            pos_col = [new_tbl.iloc[0:i_size,0:i_size].min().idxmin()]
                            pos_line = [new_tbl.iloc[0:i_size,0:i_size].idxmin()[new_tbl.iloc[0:i_size,0:i_size].min().idxmin()]]
                            new_tbl = new_tbl.drop(columns = pos_col)
                            new_tbl = new_tbl.drop(pos_line)
                            new_tbl = new_tbl.reset_index(drop=True)
                            i_size = min(new_tbl.shape)
                        else:
                            i_size = 0
                            
                    min_local = np.mean(local_store)
                    
                    stored_min.append(min_local)
                char_result.append(np.mean(stored_min))
        # print(full_results)
        char_result = 1 - char_result/max(char_result)
        return char_result, full_results

