import os
import sys

# Suppress as many warnings as possible
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# from tensorflow.python.util import deprecation
# deprecation._PRINT_DEPRECATION_WARNINGS = False
# import tensorflow as tf
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from ckiptagger import data_utils, construct_dictionary, WS, POS, NER    #記得pip install

def main():
    # Download data
    #data_utils.download_data("./")    #第一次執行需要這行 把前面#弄掉
    
    # Load model
    ws = WS("./data")
    pos = POS("./data")
    ner = NER("./data")

    word_to_weight = {"橋本有菜": 1,}             #因為CKIP不認識橋本有菜，所以要教
    dictionary = construct_dictionary(word_to_weight)
    
    txt = open('./input.txt',"r",encoding="utf-8") #輸入文字檔
    sentence_list = []
    for line in txt:
        line=line.strip('\n')            #讀取文件 並變成CKIP吃的list
        sentence_list.append(line) 
    print(sentence_list)
    
    # Run WS-POS-NER pipeline
    '''sentence_list = [
        "傅達仁今將執行安樂死，卻突然爆出自己20年前遭緯來體育台封殺，他不懂自己哪裡得罪到電視台。",
        "美國參議院針對今天總統布什所提名的勞工部長趙小蘭展開認可聽證會，預料她將會很順利通過參議院支持，成為該國有史以來第一位的華裔女性內閣成員。",
        "",
        "土地公有政策?？還是土地婆有政策。.",
        "… 你確定嗎… 不要再騙了……",
        "最多容納59,000個人,或5.9萬人,再多就不行了.這是環評的結論.",
        "科長說:1,坪數對人數為1:3。2,可以再增加。",
    ]'''
    #word_sentence_list = ws(sentence_list)
    word_sentence_list = ws(sentence_list, recommend_dictionary=dictionary) #要認識橋本就套用這行有字典的，不想認識就套上一行
    pos_sentence_list = pos(word_sentence_list)
    entity_sentence_list = ner(word_sentence_list, pos_sentence_list)
    
    # Release model
    del ws
    del pos         #我們放上去雲端之後應該不用release
    del ner
    
    # Show results
    output = open('output.txt', 'w', encoding='utf-8')    #輸出文字檔
    def print_word_pos_sentence(word_sentence, pos_sentence):
        assert len(word_sentence) == len(pos_sentence)
        for word, pos in zip(word_sentence, pos_sentence):
            #print(f"{word}", end="\u3000")
            output.write(f"{word}"+" ")                             #output的重點在這
        #print()
        output.write('\n')
    
    for i, sentence in enumerate(sentence_list):
        #print()
        #print(f"'{sentence}'")
        print_word_pos_sentence(word_sentence_list[i],  pos_sentence_list[i])
        
    
    
if __name__ == "__main__":
    main()
    sys.exit()