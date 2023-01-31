import pandas as pd
  
#Dir cartelle
csv = "validation_coco.csv"              
file_csv =pd.read_csv(csv)         # ID IMMAGINI
        
       
#Carico la lista
id_imm_list = file_csv["id"]
questions_list = file_csv["Domande"]
cocolist = file_csv["immagine_id"]





for i in range(len(id_imm_list)):
    list_ = []
    for original_string in questions_list[i].split(','):
        
        characters_to_remove = "[]'"
        
        new_string = original_string
        
        for character in characters_to_remove:
            new_string = new_string.replace(character, "")
        
        #print("Strig " , new_string) 
        list_ +=[new_string]

    lista3 = [list_]
    print(lista3)
        



        

    

    