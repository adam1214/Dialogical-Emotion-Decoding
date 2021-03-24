import joblib
import numpy as np

if __name__ == '__main__':
    emo_dict = joblib.load('emo_all_iemocap.pkl')
    dialogs = joblib.load('dialog_iemocap.pkl')
    out_dict = joblib.load('outputs.pkl')
    emo_list = ['sur', 'fru', 'oth', 'dis', 'fea', 'xxx']
    replace_emo_list = []
    for E in emo_list:
        ang_cnt = 0
        hap_cnt = 0
        neu_cnt = 0
        sad_cnt = 0
        replace_candi_emo = np.zeros(shape=(1,4))
        for dialog in dialogs.values():
            for utt in dialog:
                if emo_dict[utt] == E:
                    if np.argmax(out_dict[utt]) == 0:
                        ang_cnt += 1
                    elif np.argmax(out_dict[utt]) == 1:
                        hap_cnt += 1
                    elif np.argmax(out_dict[utt]) == 2:
                        neu_cnt += 1
                    elif np.argmax(out_dict[utt]) == 3:
                        sad_cnt += 1
        print('=======================================================')
        print(E)
        print('ang:', ang_cnt / (ang_cnt + hap_cnt + neu_cnt + sad_cnt) )
        print('hap:', hap_cnt / (ang_cnt + hap_cnt + neu_cnt + sad_cnt) )
        print('neu:', neu_cnt / (ang_cnt + hap_cnt + neu_cnt + sad_cnt) )
        print('sad:', sad_cnt / (ang_cnt + hap_cnt + neu_cnt + sad_cnt) )
        
        replace_candi_emo[0][0] = ang_cnt / (ang_cnt + hap_cnt + neu_cnt + sad_cnt)
        replace_candi_emo[0][1] = hap_cnt / (ang_cnt + hap_cnt + neu_cnt + sad_cnt)
        replace_candi_emo[0][2] = neu_cnt / (ang_cnt + hap_cnt + neu_cnt + sad_cnt)
        replace_candi_emo[0][3] = sad_cnt / (ang_cnt + hap_cnt + neu_cnt + sad_cnt)
        if np.argmax(replace_candi_emo) == 0:
            replace_emo_list.append('ang')
        elif np.argmax(replace_candi_emo) == 1:
            replace_emo_list.append('hap')
        elif np.argmax(replace_candi_emo) == 2:
            replace_emo_list.append('neu')
        elif np.argmax(replace_candi_emo) == 3:
            replace_emo_list.append('sad')
    print(replace_emo_list)

    for utt in emo_dict:
        for index in range(0, len(emo_list), 1):
            if emo_dict[utt] == emo_list[index]:
                emo_dict[utt] = replace_emo_list[index]
                break
    joblib.dump(emo_dict, 'C2C_4emo_all_iemocap.pkl') #class to class mapping according to pretrained classifier
    #################################################################################################################
    emo_dict = joblib.load('emo_all_iemocap.pkl')
    for utt in emo_dict:
        for index in range(0, len(emo_list), 1):
            if emo_dict[utt] == emo_list[index]:
                if np.argmax(out_dict[utt]) == 0:
                    emo_dict[utt] = 'ang'
                elif np.argmax(out_dict[utt]) == 1:
                    emo_dict[utt] = 'hap'
                elif np.argmax(out_dict[utt]) == 2:
                    emo_dict[utt] = 'neu'
                elif np.argmax(out_dict[utt]) == 3:
                    emo_dict[utt] = 'sad'
                break
    joblib.dump(emo_dict, 'U2U_4emo_all_iemocap.pkl') #utterance to utterance mapping according to pretrained classifier
    #################################################################################################################
    dialogs_edit = joblib.load('dialog_iemocap.pkl')
    dialogs = joblib.load('dialog_iemocap.pkl')
    emo_dict = joblib.load('emo_all_iemocap.pkl')
    for dialog in dialogs.values():
        for utt in dialog:
            if emo_dict[utt] != 'ang' and emo_dict[utt] != 'hap' and emo_dict[utt] != 'neu' and emo_dict[utt] != 'sad':
                dialogs_edit[utt[:-5]].remove(utt)
    joblib.dump(dialogs_edit, 'dialog_4emo_iemocap.pkl')