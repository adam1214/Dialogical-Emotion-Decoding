import numpy as np
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score

def split_dialog(dialogs):
  """Split utterances in a dialog into a set of speaker's utternaces in that dialog.
     See eq (5) in the paper.
  Arg:
    dialogs: dict, for example, utterances of two speakers in dialog_01: 
            {dialog_01: [utt_spk01_1, utt_spk02_1, utt_spk01_2, ...]}.
  Return:
    spk_dialogs: dict, a collection of speakers' utterances in dialogs. for example:
            {dialog_01_spk01: [utt_spk01_1, utt_spk01_2, ...],
             dialog_01_spk02: [utt_spk02_1, utt_spk02_2, ...]}
  """

  spk_dialogs = {}
  for dialog_id in dialogs.keys():
    spk_dialogs[dialog_id+'_M'] = []
    spk_dialogs[dialog_id+'_F'] = []
    for utt_id in dialogs[dialog_id]:
      if utt_id[-4] == 'M':
        spk_dialogs[dialog_id+'_M'].append(utt_id)
      elif utt_id[-4] == 'F':
        spk_dialogs[dialog_id+'_F'].append(utt_id)

  return spk_dialogs

def transition_bias(spk_dialogs, emo, val=None):
  """Estimate the transition bias of emotion. See eq (5) in the paper.
  Args:
    spk_dialogs: dict, a collection of speakers' utterances in dialogs. for example:
    emo: dict, map from utt_id to emotion state.
    val: str, validation session. If given, calculate the trainsition bias except 
         dialogs in the validation session. For example, 'Ses01'.

  Returns: 
    bias: p_0 in eq (4).
  """
  transit_num = 0
  total_transit = 0
  count = 0
  num = 0
  for dialog_id in spk_dialogs.values():
    if val and val == dialog_id[0][:5]:
      continue

    for entry in range(len(dialog_id) - 1):
      transit_num += (emo[dialog_id[entry]] != emo[dialog_id[entry + 1]])
    total_transit += (len(dialog_id) - 1)

  bias = (transit_num + 1) / total_transit

  return bias, total_transit

def get_val_bias(dialog, emo_dict):
    """Get p_0 estimated from training sessions."""

    session = ['Ses01', 'Ses02', 'Ses03', 'Ses04', 'Ses05']
    bias_dict = {}
    for i in range(len(session)):
      val = session[i]
      train_sessions = session[:i] + session[i+1:]
      p_0, _ = transition_bias(dialog, emo_dict, val)
      print("Transition bias of { %s }: %.3f" % (' ,'.join(train_sessions), p_0))
      bias_dict[val] = p_0

    return bias_dict

def softmax(x):
  """Compute the softmax of vector x."""
  exp_x = np.exp(x)
  softmax_x = exp_x / np.sum(exp_x)
  return softmax_x 

def emo_trans_prob_BI_need_softmax(emo_dict, dialogs, add_one_smooth_or_not, val=None):
    # only estimate anger, happiness, neutral, sadness
    total_transit = 0

    ang2ang = 0
    ang2hap = 0
    ang2neu = 0
    ang2sad = 0
    
    hap2ang = 0
    hap2hap = 0
    hap2neu = 0
    hap2sad = 0

    neu2ang = 0
    neu2hap = 0
    neu2neu = 0
    neu2sad = 0

    sad2ang = 0
    sad2hap = 0
    sad2neu = 0
    sad2sad = 0

    pre_emo = ''
    pre_dialog_id = ''
    last_dialog_last_utt_emo = ''

    for dialog in dialogs.values():
        for utt in dialog:
            last_dialog_last_utt_emo = emo_dict[utt]
            dialog_id = utt[0:-5]
            #print(dialog_id)
            if val and val == dialog_id[0:5]:
                continue

            if emo_dict[utt] != 'ang' and emo_dict[utt] != 'hap' and emo_dict[utt] != 'neu' and emo_dict[utt] != 'sad': 
                # only estimate anger, happiness, neutral, sadness
                pre_dialog_id = dialog_id
                continue

            if pre_emo == '' and pre_dialog_id == '': #begining of the traversal
                pre_emo = emo_dict[utt]
                pre_dialog_id = dialog_id
                #total_transit += 1
                continue

            if pre_dialog_id != dialog_id: #new dialog
                #total_transit -= 1
                pre_emo = ''

            if pre_emo == 'ang' and emo_dict[utt] == 'ang':
                ang2ang += 1
                total_transit += 1
            if pre_emo == 'ang' and emo_dict[utt] == 'hap':
                ang2hap += 1
                total_transit += 1
            if pre_emo == 'ang' and emo_dict[utt] == 'neu':
                ang2neu += 1
                total_transit += 1
            if pre_emo == 'ang' and emo_dict[utt] == 'sad':
                ang2sad += 1
                total_transit += 1

            if pre_emo == 'hap' and emo_dict[utt] == 'ang':
                hap2ang += 1
                total_transit += 1
            if pre_emo == 'hap' and emo_dict[utt] == 'hap':
                hap2hap += 1
                total_transit += 1
            if pre_emo == 'hap' and emo_dict[utt] == 'neu':
                hap2neu += 1
                total_transit += 1
            if pre_emo == 'hap' and emo_dict[utt] == 'sad':
                hap2sad += 1
                total_transit += 1

            if pre_emo == 'neu' and emo_dict[utt] == 'ang':
                neu2ang += 1
                total_transit += 1
            if pre_emo == 'neu' and emo_dict[utt] == 'hap':
                neu2hap += 1
                total_transit += 1
            if pre_emo == 'neu' and emo_dict[utt] == 'neu':
                neu2neu += 1
                total_transit += 1
            if pre_emo == 'neu' and emo_dict[utt] == 'sad':
                neu2sad += 1
                total_transit += 1

            if pre_emo == 'sad' and emo_dict[utt] == 'ang':
                sad2ang += 1
                total_transit += 1
            if pre_emo == 'sad' and emo_dict[utt] == 'hap':
                sad2hap += 1
                total_transit += 1
            if pre_emo == 'sad' and emo_dict[utt] == 'neu':
                sad2neu += 1
                total_transit += 1
            if pre_emo == 'sad' and emo_dict[utt] == 'sad':
                sad2sad += 1
                total_transit += 1
            
            pre_dialog_id = dialog_id
            pre_emo = emo_dict[utt]
    #if last_dialog_last_utt_emo == 'ang' or last_dialog_last_utt_emo == 'hap' or last_dialog_last_utt_emo == 'neu' or last_dialog_last_utt_emo == 'sad':
        #total_transit -= 1
    print("before softmax:")
    print(ang2ang/total_transit+ang2hap/total_transit+ang2neu/total_transit+ang2sad/total_transit+hap2ang/total_transit+hap2hap/total_transit+hap2neu/total_transit+hap2sad/total_transit+neu2ang/total_transit+neu2hap/total_transit+neu2neu/total_transit+neu2sad/total_transit+sad2ang/total_transit+sad2hap/total_transit+sad2neu/total_transit+sad2sad/total_transit)
    a = softmax([ang2ang/total_transit, ang2hap/total_transit, ang2neu/total_transit, ang2sad/total_transit])
    h = softmax([hap2ang/total_transit, hap2hap/total_transit, hap2neu/total_transit, hap2sad/total_transit])
    n = softmax([neu2ang/total_transit, neu2hap/total_transit, neu2neu/total_transit, neu2sad/total_transit])
    s = softmax([sad2ang/total_transit, sad2hap/total_transit, sad2neu/total_transit, sad2sad/total_transit])
    return {'a2a':a[0], 'a2h':a[1], 'a2n':a[2], 'a2s':a[3], \
                    'h2a':h[0], 'h2h':h[1], 'h2n':h[2], 'h2s':h[3], \
                    'n2a':n[0], 'n2h':n[1], 'n2n':n[2], 'n2s':n[3], \
                    's2a':s[0], 's2h':s[1], 's2n':s[2], 's2s':s[3]}

def emo_trans_prob_BI_without_softmax(emo_dict, dialogs, val=None):
    # only estimate anger, happiness, neutral, sadness
    a2 = 0
    h2 = 0
    n2 = 0
    s2 = 0

    ang2ang = 0
    ang2hap = 0
    ang2neu = 0
    ang2sad = 0
    
    hap2ang = 0
    hap2hap = 0
    hap2neu = 0
    hap2sad = 0

    neu2ang = 0
    neu2hap = 0
    neu2neu = 0
    neu2sad = 0

    sad2ang = 0
    sad2hap = 0
    sad2neu = 0
    sad2sad = 0

    pre_emo = ''
    pre_dialog_id = ''
    last_dialog_last_utt_emo = ''

    for dialog in dialogs.values():
        for utt in dialog:
            last_dialog_last_utt_emo = emo_dict[utt]
            dialog_id = utt[0:-5]
            #print(dialog_id)
            if val and val == dialog_id[0:5]:
                continue

            if emo_dict[utt] != 'ang' and emo_dict[utt] != 'hap' and emo_dict[utt] != 'neu' and emo_dict[utt] != 'sad': 
                # only estimate anger, happiness, neutral, sadness
                pre_dialog_id = dialog_id
                continue

            if pre_emo == '' and pre_dialog_id == '': #begining of the traversal
                pre_emo = emo_dict[utt]
                pre_dialog_id = dialog_id
                '''
                if pre_emo == 'ang':
                    a2 += 1
                elif pre_emo == 'hap':
                    h2 += 1
                elif pre_emo == 'neu':
                    n2 += 1
                else:
                    s2 += 1
                '''
                continue

            if pre_dialog_id != dialog_id: #new dialog
                '''
                if pre_emo == 'ang':
                    a2 -= 1
                elif pre_emo == 'hap':
                    h2 -= 1
                elif pre_emo == 'neu':
                    n2 -= 1
                else:
                    s2 -= 1
                '''
                pre_emo = ''

            if pre_emo == 'ang' and emo_dict[utt] == 'ang':
                ang2ang += 1
                a2 += 1
            if pre_emo == 'ang' and emo_dict[utt] == 'hap':
                ang2hap += 1
                a2 += 1
            if pre_emo == 'ang' and emo_dict[utt] == 'neu':
                ang2neu += 1
                a2 += 1
            if pre_emo == 'ang' and emo_dict[utt] == 'sad':
                ang2sad += 1
                a2 += 1

            if pre_emo == 'hap' and emo_dict[utt] == 'ang':
                hap2ang += 1
                h2 += 1
            if pre_emo == 'hap' and emo_dict[utt] == 'hap':
                hap2hap += 1
                h2 += 1
            if pre_emo == 'hap' and emo_dict[utt] == 'neu':
                hap2neu += 1
                h2 += 1
            if pre_emo == 'hap' and emo_dict[utt] == 'sad':
                hap2sad += 1
                h2 += 1

            if pre_emo == 'neu' and emo_dict[utt] == 'ang':
                neu2ang += 1
                n2 += 1
            if pre_emo == 'neu' and emo_dict[utt] == 'hap':
                neu2hap += 1
                n2 += 1
            if pre_emo == 'neu' and emo_dict[utt] == 'neu':
                neu2neu += 1
                n2 += 1
            if pre_emo == 'neu' and emo_dict[utt] == 'sad':
                neu2sad += 1
                n2 += 1

            if pre_emo == 'sad' and emo_dict[utt] == 'ang':
                sad2ang += 1
                s2 += 1
            if pre_emo == 'sad' and emo_dict[utt] == 'hap':
                sad2hap += 1
                s2 += 1
            if pre_emo == 'sad' and emo_dict[utt] == 'neu':
                sad2neu += 1
                s2 += 1
            if pre_emo == 'sad' and emo_dict[utt] == 'sad':
                sad2sad += 1
                s2 += 1
            
            pre_dialog_id = dialog_id
            pre_emo = emo_dict[utt]
            '''
            if pre_emo == 'ang':
                a2 += 1
            elif pre_emo == 'hap':
                h2 += 1
            elif pre_emo == 'neu':
                n2 += 1
            else:
                s2 += 1
            '''
    '''
    if last_dialog_last_utt_emo == 'ang':
        a2 -= 1
    elif last_dialog_last_utt_emo == 'hap':
        h2 -= 1
    elif last_dialog_last_utt_emo == 'neu':
        n2 -= 1
    elif last_dialog_last_utt_emo == 'sad':
        s2 -= 1
    '''
    print(ang2ang/a2+ang2hap/a2+ang2neu/a2+ang2sad/a2)
    print(hap2ang/h2+hap2hap/h2+hap2neu/h2+hap2sad/h2)
    print(neu2ang/n2+neu2hap/n2+neu2neu/n2+neu2sad/n2)
    print(sad2ang/s2+sad2hap/s2+sad2neu/s2+sad2sad/s2)
    print('=============================================')
    return {'a2a':ang2ang/a2, 'a2h':ang2hap/a2, 'a2n':ang2neu/a2, 'a2s':ang2sad/a2, \
            'h2a':hap2ang/h2, 'h2h':hap2hap/h2, 'h2n':hap2neu/h2, 'h2s':hap2sad/h2, \
            'n2a':neu2ang/n2, 'n2h':neu2hap/n2, 'n2n':neu2neu/n2, 'n2s':neu2sad/n2, \
            's2a':sad2ang/s2, 's2h':sad2hap/s2, 's2n':sad2neu/s2, 's2s':sad2sad/s2  }

def emo_trans_prob_TRI_need_softmax(emo_dict, dialogs, add_one_smooth_or_not ,val=None):
    # only estimate anger, happiness, neutral, sadness
    total_transit = 0

    aaa = 0
    aah = 0
    aan = 0
    aas = 0
    aha = 0
    ahh = 0
    ahn = 0
    ahs = 0
    ana = 0
    anh = 0
    ann = 0
    ans = 0
    asa = 0
    ash = 0
    asn = 0
    ass = 0

    haa = 0
    hah = 0
    han = 0
    has = 0
    hha = 0
    hhh = 0
    hhn = 0
    hhs = 0
    hna = 0
    hnh = 0
    hnn = 0
    hns = 0
    hsa = 0
    hsh = 0
    hsn = 0
    hss = 0

    naa = 0
    nah = 0
    nan = 0
    nas = 0
    nha = 0
    nhh = 0
    nhn = 0
    nhs = 0
    nna = 0
    nnh = 0
    nnn = 0
    nns = 0
    nsa = 0
    nsh = 0
    nsn = 0
    nss = 0

    saa = 0
    sah = 0
    san = 0
    sas = 0
    sha = 0
    shh = 0
    shn = 0
    shs = 0
    sna = 0
    snh = 0
    snn = 0
    sns = 0
    ssa = 0
    ssh = 0
    ssn = 0
    sss = 0

    pre_emo = ''
    pre_pre_emo = ''
    pre_dialog_id = ''
    last_dialog_last_utt_emo = ''

    for dialog in dialogs.values():
        for utt in dialog:
            last_dialog_last_utt_emo = emo_dict[utt]
            dialog_id = utt[0:-5]
            #print(dialog_id)
            if val and val == dialog_id[0:5]:
                continue

            if emo_dict[utt] != 'ang' and emo_dict[utt] != 'hap' and emo_dict[utt] != 'neu' and emo_dict[utt] != 'sad': 
                # only estimate anger, happiness, neutral, sadness
                pre_dialog_id = dialog_id
                continue

            if pre_emo == '' and pre_dialog_id == '': #begining of the traversal
                pre_emo = emo_dict[utt]
                pre_dialog_id = dialog_id
                #total_transit += 1
                continue

            if pre_emo != '' and pre_pre_emo == '': #after one step of new dialog traversal
                pre_pre_emo = pre_emo
                pre_emo = emo_dict[utt]
                pre_dialog_id = dialog_id
                continue

            if pre_dialog_id != dialog_id: #new dialog
                #total_transit -= 1
                pre_emo = ''
                pre_pre_emo = ''
            
            if pre_pre_emo == 'ang' and pre_emo == 'ang' and emo_dict[utt] == 'ang':
                aaa += 1
                total_transit += 1
            if pre_pre_emo == 'ang' and pre_emo == 'ang' and emo_dict[utt] == 'hap':
                aah += 1
                total_transit += 1
            if pre_pre_emo == 'ang' and pre_emo == 'ang' and emo_dict[utt] == 'neu':
                aan += 1
                total_transit += 1
            if pre_pre_emo == 'ang' and pre_emo == 'ang' and emo_dict[utt] == 'sad':
                aas += 1
                total_transit += 1

            if pre_pre_emo == 'ang' and pre_emo == 'hap' and emo_dict[utt] == 'ang':
                aha += 1
                total_transit += 1
            if pre_pre_emo == 'ang' and pre_emo == 'hap' and emo_dict[utt] == 'hap':
                ahh += 1
                total_transit += 1
            if pre_pre_emo == 'ang' and pre_emo == 'hap' and emo_dict[utt] == 'neu':
                ahn += 1
                total_transit += 1
            if pre_pre_emo == 'ang' and pre_emo == 'hap' and emo_dict[utt] == 'sad':
                ahs += 1
                total_transit += 1

            if pre_pre_emo == 'ang' and pre_emo == 'neu' and emo_dict[utt] == 'ang':
                ana += 1
                total_transit += 1
            if pre_pre_emo == 'ang' and pre_emo == 'neu' and emo_dict[utt] == 'hap':
                anh += 1
                total_transit += 1
            if pre_pre_emo == 'ang' and pre_emo == 'neu' and emo_dict[utt] == 'neu':
                ann += 1
                total_transit += 1
            if pre_pre_emo == 'ang' and pre_emo == 'neu' and emo_dict[utt] == 'sad':
                ans += 1
                total_transit += 1

            if pre_pre_emo == 'ang' and pre_emo == 'sad' and emo_dict[utt] == 'ang':
                asa += 1
                total_transit += 1
            if pre_pre_emo == 'ang' and pre_emo == 'sad' and emo_dict[utt] == 'hap':
                ash += 1
                total_transit += 1
            if pre_pre_emo == 'ang' and pre_emo == 'sad' and emo_dict[utt] == 'neu':
                asn += 1
                total_transit += 1
            if pre_pre_emo == 'ang' and pre_emo == 'sad' and emo_dict[utt] == 'sad':
                ass += 1
                total_transit += 1
            #########################################################################
            if pre_pre_emo == 'hap' and pre_emo == 'ang' and emo_dict[utt] == 'ang':
                haa += 1
                total_transit += 1
            if pre_pre_emo == 'hap' and pre_emo == 'ang' and emo_dict[utt] == 'hap':
                hah += 1
                total_transit += 1
            if pre_pre_emo == 'hap' and pre_emo == 'ang' and emo_dict[utt] == 'neu':
                han += 1
                total_transit += 1
            if pre_pre_emo == 'hap' and pre_emo == 'ang' and emo_dict[utt] == 'sad':
                has += 1
                total_transit += 1

            if pre_pre_emo == 'hap' and pre_emo == 'hap' and emo_dict[utt] == 'ang':
                hha += 1
                total_transit += 1
            if pre_pre_emo == 'hap' and pre_emo == 'hap' and emo_dict[utt] == 'hap':
                hhh += 1
                total_transit += 1
            if pre_pre_emo == 'hap' and pre_emo == 'hap' and emo_dict[utt] == 'neu':
                hhn += 1
                total_transit += 1
            if pre_pre_emo == 'hap' and pre_emo == 'hap' and emo_dict[utt] == 'sad':
                hhs += 1
                total_transit += 1

            if pre_pre_emo == 'hap' and pre_emo == 'neu' and emo_dict[utt] == 'ang':
                hna += 1
                total_transit += 1
            if pre_pre_emo == 'hap' and pre_emo == 'neu' and emo_dict[utt] == 'hap':
                hnh += 1
                total_transit += 1
            if pre_pre_emo == 'hap' and pre_emo == 'neu' and emo_dict[utt] == 'neu':
                hnn += 1
                total_transit += 1
            if pre_pre_emo == 'hap' and pre_emo == 'neu' and emo_dict[utt] == 'sad':
                hns += 1
                total_transit += 1

            if pre_pre_emo == 'hap' and pre_emo == 'sad' and emo_dict[utt] == 'ang':
                hsa += 1
                total_transit += 1
            if pre_pre_emo == 'hap' and pre_emo == 'sad' and emo_dict[utt] == 'hap':
                hsh += 1
                total_transit += 1
            if pre_pre_emo == 'hap' and pre_emo == 'sad' and emo_dict[utt] == 'neu':
                hsn += 1
                total_transit += 1
            if pre_pre_emo == 'hap' and pre_emo == 'sad' and emo_dict[utt] == 'sad':
                hss += 1
                total_transit += 1
            #########################################################################
            if pre_pre_emo == 'neu' and pre_emo == 'ang' and emo_dict[utt] == 'ang':
                naa += 1
                total_transit += 1
            if pre_pre_emo == 'neu' and pre_emo == 'ang' and emo_dict[utt] == 'hap':
                nah += 1
                total_transit += 1
            if pre_pre_emo == 'neu' and pre_emo == 'ang' and emo_dict[utt] == 'neu':
                nan += 1
                total_transit += 1
            if pre_pre_emo == 'neu' and pre_emo == 'ang' and emo_dict[utt] == 'sad':
                nas += 1
                total_transit += 1

            if pre_pre_emo == 'neu' and pre_emo == 'hap' and emo_dict[utt] == 'ang':
                nha += 1
                total_transit += 1
            if pre_pre_emo == 'neu' and pre_emo == 'hap' and emo_dict[utt] == 'hap':
                nhh += 1
                total_transit += 1
            if pre_pre_emo == 'neu' and pre_emo == 'hap' and emo_dict[utt] == 'neu':
                nhn += 1
                total_transit += 1
            if pre_pre_emo == 'neu' and pre_emo == 'hap' and emo_dict[utt] == 'sad':
                nhs += 1
                total_transit += 1

            if pre_pre_emo == 'neu' and pre_emo == 'neu' and emo_dict[utt] == 'ang':
                nna += 1
                total_transit += 1
            if pre_pre_emo == 'neu' and pre_emo == 'neu' and emo_dict[utt] == 'hap':
                nnh += 1
                total_transit += 1
            if pre_pre_emo == 'neu' and pre_emo == 'neu' and emo_dict[utt] == 'neu':
                nnn += 1
                total_transit += 1
            if pre_pre_emo == 'neu' and pre_emo == 'neu' and emo_dict[utt] == 'sad':
                nns += 1
                total_transit += 1

            if pre_pre_emo == 'neu' and pre_emo == 'sad' and emo_dict[utt] == 'ang':
                nsa += 1
                total_transit += 1
            if pre_pre_emo == 'neu' and pre_emo == 'sad' and emo_dict[utt] == 'hap':
                nsh += 1
                total_transit += 1
            if pre_pre_emo == 'neu' and pre_emo == 'sad' and emo_dict[utt] == 'neu':
                nsn += 1
                total_transit += 1
            if pre_pre_emo == 'neu' and pre_emo == 'sad' and emo_dict[utt] == 'sad':
                nss += 1          
                total_transit += 1
            #########################################################################
            if pre_pre_emo == 'sad' and pre_emo == 'ang' and emo_dict[utt] == 'ang':
                saa += 1
                total_transit += 1
            if pre_pre_emo == 'sad' and pre_emo == 'ang' and emo_dict[utt] == 'hap':
                sah += 1
                total_transit += 1
            if pre_pre_emo == 'sad' and pre_emo == 'ang' and emo_dict[utt] == 'neu':
                san += 1
                total_transit += 1
            if pre_pre_emo == 'sad' and pre_emo == 'ang' and emo_dict[utt] == 'sad':
                sas += 1
                total_transit += 1

            if pre_pre_emo == 'sad' and pre_emo == 'hap' and emo_dict[utt] == 'ang':
                sha += 1
                total_transit += 1
            if pre_pre_emo == 'sad' and pre_emo == 'hap' and emo_dict[utt] == 'hap':
                shh += 1
                total_transit += 1
            if pre_pre_emo == 'sad' and pre_emo == 'hap' and emo_dict[utt] == 'neu':
                shn += 1
                total_transit += 1
            if pre_pre_emo == 'sad' and pre_emo == 'hap' and emo_dict[utt] == 'sad':
                shs += 1
                total_transit += 1

            if pre_pre_emo == 'sad' and pre_emo == 'neu' and emo_dict[utt] == 'ang':
                sna += 1
                total_transit += 1
            if pre_pre_emo == 'sad' and pre_emo == 'neu' and emo_dict[utt] == 'hap':
                snh += 1
                total_transit += 1
            if pre_pre_emo == 'sad' and pre_emo == 'neu' and emo_dict[utt] == 'neu':
                snn += 1
                total_transit += 1
            if pre_pre_emo == 'sad' and pre_emo == 'neu' and emo_dict[utt] == 'sad':
                sns += 1
                total_transit += 1

            if pre_pre_emo == 'sad' and pre_emo == 'sad' and emo_dict[utt] == 'ang':
                ssa += 1
                total_transit += 1
            if pre_pre_emo == 'sad' and pre_emo == 'sad' and emo_dict[utt] == 'hap':
                ssh += 1
                total_transit += 1
            if pre_pre_emo == 'sad' and pre_emo == 'sad' and emo_dict[utt] == 'neu':
                ssn += 1
                total_transit += 1
            if pre_pre_emo == 'sad' and pre_emo == 'sad' and emo_dict[utt] == 'sad':
                sss += 1
                total_transit += 1
   
            pre_dialog_id = dialog_id
            pre_pre_emo = pre_emo
            pre_emo = emo_dict[utt]
    #if last_dialog_last_utt_emo == 'ang' or last_dialog_last_utt_emo == 'hap' or last_dialog_last_utt_emo == 'neu' or last_dialog_last_utt_emo == 'sad':
        #total_transit -= 1

    if add_one_smooth_or_not == 1:
        total_transit += 64

        aaa += 1
        aah += 1
        aan += 1
        aas += 1
        aha += 1
        ahh += 1
        ahn += 1
        ahs += 1
        ana += 1
        anh += 1
        ann += 1
        ans += 1
        asa += 1
        ash += 1
        asn += 1
        ass += 1

        haa += 1
        hah += 1
        han += 1
        has += 1
        hha += 1
        hhh += 1
        hhn += 1
        hhs += 1
        hna += 1
        hnh += 1
        hnn += 1
        hns += 1
        hsa += 1
        hsh += 1
        hsn += 1
        hss += 1

        naa += 1
        nah += 1
        nan += 1
        nas += 1
        nha += 1
        nhh += 1
        nhn += 1
        nhs += 1
        nna += 1
        nnh += 1
        nnn += 1
        nns += 1
        nsa += 1
        nsh += 1
        nsn += 1
        nss += 1

        saa += 1
        sah += 1
        san += 1
        sas += 1
        sha += 1
        shh += 1
        shn += 1
        shs += 1
        sna += 1
        snh += 1
        snn += 1
        sns += 1
        ssa += 1
        ssh += 1
        ssn += 1
        sss += 1
    print("before softmax:")
    print( (aaa+aah+aan+aas+aha+ahh+ahn+ahs+ana+anh+ann+ans+asa+ash+asn+ass+haa+hah+han+has+hha+hhh+hhn+hhs+hna+hnh+hnn+hns+hsa+hsh+hsn+hss+naa+nah+nan+nas+nha+nhh+nhn+nhs+nna+nnh+nnn+nns+nsa+nsh+nsn+nss+saa+sah+san+sas+sha+shh+shn+shs+sna+snh+snn+sns+ssa+ssh+ssn+sss)/total_transit )
    '''
    print('aaa', aaa)
    print('aah', aah)
    print('aan', aan)
    print('aas', aas)
    print('aha', aha)
    print('ahh', ahh)
    print('ahn', ahn)
    print('ahs', ahs)
    print('ana', ana)
    print('anh', anh)
    print('ann', ann)
    print('ans', ans)
    print('asa', asa)
    print('ash', ash)
    print('asn', asn)
    print('ass', ass)
    #################
    print('haa', haa)
    print('hah', hah)
    print('han', han)
    print('has', has)
    print('hha', hha)
    print('hhh', hhh)
    print('hhn', hhn)
    print('hhs', hhs)
    print('hna', hna)
    print('hnh', hnh)
    print('hnn', hnn)
    print('hns', hns)
    print('hsa', hsa)
    print('hsh', hsh)
    print('hsn', hsn)
    print('hss', hss)
    #################
    print('naa', naa)
    print('nah', nah)
    print('nan', nan)
    print('nas', nas)
    print('nha', nha)
    print('nhh', nhh)
    print('nhn', nhn)
    print('nhs', nhs)
    print('nna', nna)
    print('nnh', nnh)
    print('nnn', nnn)
    print('nns', nns)
    print('nsa', nsa)
    print('nsh', nsh)
    print('nsn', nsn)
    print('nss', nss)
    #################
    print('saa', saa)
    print('sah', sah)
    print('san', san)
    print('sas', sas)
    print('sha', sha)
    print('shh', shh)
    print('shn', shn)
    print('shs', shs)
    print('sna', sna)
    print('snh', snh)
    print('snn', snn)
    print('sns', sns)
    print('ssa', ssa)
    print('ssh', ssh)
    print('ssn', ssn)
    print('sss', sss)
    print('total_transit', total_transit)
    '''
    a_a = softmax([aaa/total_transit, aah/total_transit, aan/total_transit, aas/total_transit])
    a_h = softmax([aha/total_transit, ahh/total_transit, ahn/total_transit, ahs/total_transit])
    a_n = softmax([ana/total_transit, anh/total_transit, ann/total_transit, ans/total_transit])
    a_s = softmax([asa/total_transit, ash/total_transit, asn/total_transit, ass/total_transit])
    ###########################################################################################
    h_a = softmax([haa/total_transit, hah/total_transit, han/total_transit, has/total_transit])
    h_h = softmax([hha/total_transit, hhh/total_transit, hhn/total_transit, hhs/total_transit])
    h_n = softmax([hna/total_transit, hnh/total_transit, hnn/total_transit, hns/total_transit])
    h_s = softmax([hsa/total_transit, hsh/total_transit, hsn/total_transit, hss/total_transit])
    ###########################################################################################
    n_a = softmax([naa/total_transit, nah/total_transit, nan/total_transit, nas/total_transit])
    n_h = softmax([nha/total_transit, nhh/total_transit, nhn/total_transit, nhs/total_transit])
    n_n = softmax([nna/total_transit, nnh/total_transit, nnn/total_transit, nns/total_transit])
    n_s = softmax([nsa/total_transit, nsh/total_transit, nsn/total_transit, nss/total_transit])
    ###########################################################################################
    s_a = softmax([saa/total_transit, sah/total_transit, san/total_transit, sas/total_transit])
    s_h = softmax([sha/total_transit, shh/total_transit, shn/total_transit, shs/total_transit])
    s_n = softmax([sna/total_transit, snh/total_transit, snn/total_transit, sns/total_transit])
    s_s = softmax([ssa/total_transit, ssh/total_transit, ssn/total_transit, sss/total_transit])

    return {'aaa':a_a[0], 'aah':a_a[1], 'aan':a_a[2], 'aas':a_a[3], \
            'aha':a_h[0], 'ahh':a_h[1], 'ahn':a_h[2], 'ahs':a_h[3], \
            'ana':a_n[0], 'anh':a_n[1], 'ann':a_n[2], 'ans':a_n[3], \
            'asa':a_s[0], 'ash':a_s[1], 'asn':a_s[2], 'ass':a_s[3], \
            'haa':h_a[0], 'hah':h_a[1], 'han':h_a[2], 'has':h_a[3], \
            'hha':h_h[0], 'hhh':h_h[1], 'hhn':h_h[2], 'hhs':h_h[3], \
            'hna':h_n[0], 'hnh':h_n[1], 'hnn':h_n[2], 'hns':h_n[3], \
            'hsa':h_s[0], 'hsh':h_s[1], 'hsn':h_s[2], 'hss':h_s[3], \
            'naa':n_a[0], 'nah':n_a[1], 'nan':n_a[2], 'nas':n_a[3], \
            'nha':n_h[0], 'nhh':n_h[1], 'nhn':n_h[2], 'nhs':n_h[3], \
            'nna':n_n[0], 'nnh':n_n[1], 'nnn':n_n[2], 'nns':n_n[3], \
            'nsa':n_s[0], 'nsh':n_s[1], 'nsn':n_s[2], 'nss':n_s[3], \
            'saa':s_a[0], 'sah':s_a[1], 'san':s_a[2], 'sas':s_a[3], \
            'sha':s_h[0], 'shh':s_h[1], 'shn':s_h[2], 'shs':s_h[3], \
            'sna':s_n[0], 'snh':s_n[1], 'snn':s_n[2], 'sns':s_n[3], \
            'ssa':s_s[0], 'ssh':s_s[1], 'ssn':s_s[2], 'sss':s_s[3]  }

def emo_trans_prob_TRI_without_softmax(emo_dict, dialogs, add_one_smooth_or_not, val=None):
    # only estimate anger, happiness, neutral, sadness
    a_a = 0
    a_h = 0
    a_n = 0
    a_s = 0

    h_a = 0
    h_h = 0
    h_n = 0
    h_s = 0

    n_a = 0
    n_h = 0
    n_n = 0
    n_s = 0

    s_a = 0
    s_h = 0
    s_n = 0
    s_s = 0

    aaa = 0
    aah = 0
    aan = 0
    aas = 0
    aha = 0
    ahh = 0
    ahn = 0
    ahs = 0
    ana = 0
    anh = 0
    ann = 0
    ans = 0
    asa = 0
    ash = 0
    asn = 0
    ass = 0

    haa = 0
    hah = 0
    han = 0
    has = 0
    hha = 0
    hhh = 0
    hhn = 0
    hhs = 0
    hna = 0
    hnh = 0
    hnn = 0
    hns = 0
    hsa = 0
    hsh = 0
    hsn = 0
    hss = 0

    naa = 0
    nah = 0
    nan = 0
    nas = 0
    nha = 0
    nhh = 0
    nhn = 0
    nhs = 0
    nna = 0
    nnh = 0
    nnn = 0
    nns = 0
    nsa = 0
    nsh = 0
    nsn = 0
    nss = 0

    saa = 0
    sah = 0
    san = 0
    sas = 0
    sha = 0
    shh = 0
    shn = 0
    shs = 0
    sna = 0
    snh = 0
    snn = 0
    sns = 0
    ssa = 0
    ssh = 0
    ssn = 0
    sss = 0

    pre_emo = ''
    pre_pre_emo = ''
    pre_dialog_id = ''
    last_dialog_last_utt_emo = ''

    for dialog in dialogs.values():
        for utt in dialog:
            last_dialog_last_utt_emo = emo_dict[utt]
            dialog_id = utt[0:-5]
            #print(dialog_id)
            if val and val == dialog_id[0:5]:
                continue

            if emo_dict[utt] != 'ang' and emo_dict[utt] != 'hap' and emo_dict[utt] != 'neu' and emo_dict[utt] != 'sad': 
                # only estimate anger, happiness, neutral, sadness
                pre_dialog_id = dialog_id
                continue

            if pre_emo == '' and pre_dialog_id == '': #begining of the traversal
                pre_emo = emo_dict[utt]
                pre_dialog_id = dialog_id
                #total_transit += 1
                continue

            if pre_emo != '' and pre_pre_emo == '': #after one step of new dialog traversal
                pre_pre_emo = pre_emo
                pre_emo = emo_dict[utt]
                pre_dialog_id = dialog_id
                continue

            if pre_dialog_id != dialog_id: #new dialog
                #total_transit -= 1
                pre_emo = ''
                pre_pre_emo = ''
            
            if pre_pre_emo == 'ang' and pre_emo == 'ang' and emo_dict[utt] == 'ang':
                aaa += 1
                a_a += 1
            if pre_pre_emo == 'ang' and pre_emo == 'ang' and emo_dict[utt] == 'hap':
                aah += 1
                a_a += 1
            if pre_pre_emo == 'ang' and pre_emo == 'ang' and emo_dict[utt] == 'neu':
                aan += 1
                a_a += 1
            if pre_pre_emo == 'ang' and pre_emo == 'ang' and emo_dict[utt] == 'sad':
                aas += 1
                a_a += 1

            if pre_pre_emo == 'ang' and pre_emo == 'hap' and emo_dict[utt] == 'ang':
                aha += 1
                a_h += 1
            if pre_pre_emo == 'ang' and pre_emo == 'hap' and emo_dict[utt] == 'hap':
                ahh += 1
                a_h += 1
            if pre_pre_emo == 'ang' and pre_emo == 'hap' and emo_dict[utt] == 'neu':
                ahn += 1
                a_h += 1
            if pre_pre_emo == 'ang' and pre_emo == 'hap' and emo_dict[utt] == 'sad':
                ahs += 1
                a_h += 1

            if pre_pre_emo == 'ang' and pre_emo == 'neu' and emo_dict[utt] == 'ang':
                ana += 1
                a_n += 1
            if pre_pre_emo == 'ang' and pre_emo == 'neu' and emo_dict[utt] == 'hap':
                anh += 1
                a_n += 1
            if pre_pre_emo == 'ang' and pre_emo == 'neu' and emo_dict[utt] == 'neu':
                ann += 1
                a_n += 1
            if pre_pre_emo == 'ang' and pre_emo == 'neu' and emo_dict[utt] == 'sad':
                ans += 1
                a_n += 1

            if pre_pre_emo == 'ang' and pre_emo == 'sad' and emo_dict[utt] == 'ang':
                asa += 1
                a_s += 1
            if pre_pre_emo == 'ang' and pre_emo == 'sad' and emo_dict[utt] == 'hap':
                ash += 1
                a_s += 1
            if pre_pre_emo == 'ang' and pre_emo == 'sad' and emo_dict[utt] == 'neu':
                asn += 1
                a_s += 1
            if pre_pre_emo == 'ang' and pre_emo == 'sad' and emo_dict[utt] == 'sad':
                ass += 1
                a_s += 1
            #########################################################################
            if pre_pre_emo == 'hap' and pre_emo == 'ang' and emo_dict[utt] == 'ang':
                haa += 1
                h_a += 1
            if pre_pre_emo == 'hap' and pre_emo == 'ang' and emo_dict[utt] == 'hap':
                hah += 1
                h_a += 1
            if pre_pre_emo == 'hap' and pre_emo == 'ang' and emo_dict[utt] == 'neu':
                han += 1
                h_a += 1
            if pre_pre_emo == 'hap' and pre_emo == 'ang' and emo_dict[utt] == 'sad':
                has += 1
                h_a += 1

            if pre_pre_emo == 'hap' and pre_emo == 'hap' and emo_dict[utt] == 'ang':
                hha += 1
                h_h += 1
            if pre_pre_emo == 'hap' and pre_emo == 'hap' and emo_dict[utt] == 'hap':
                hhh += 1
                h_h += 1
            if pre_pre_emo == 'hap' and pre_emo == 'hap' and emo_dict[utt] == 'neu':
                hhn += 1
                h_h += 1
            if pre_pre_emo == 'hap' and pre_emo == 'hap' and emo_dict[utt] == 'sad':
                hhs += 1
                h_h += 1

            if pre_pre_emo == 'hap' and pre_emo == 'neu' and emo_dict[utt] == 'ang':
                hna += 1
                h_n += 1
            if pre_pre_emo == 'hap' and pre_emo == 'neu' and emo_dict[utt] == 'hap':
                hnh += 1
                h_n += 1
            if pre_pre_emo == 'hap' and pre_emo == 'neu' and emo_dict[utt] == 'neu':
                hnn += 1
                h_n += 1
            if pre_pre_emo == 'hap' and pre_emo == 'neu' and emo_dict[utt] == 'sad':
                hns += 1
                h_n += 1

            if pre_pre_emo == 'hap' and pre_emo == 'sad' and emo_dict[utt] == 'ang':
                hsa += 1
                h_s += 1
            if pre_pre_emo == 'hap' and pre_emo == 'sad' and emo_dict[utt] == 'hap':
                hsh += 1
                h_s += 1
            if pre_pre_emo == 'hap' and pre_emo == 'sad' and emo_dict[utt] == 'neu':
                hsn += 1
                h_s += 1
            if pre_pre_emo == 'hap' and pre_emo == 'sad' and emo_dict[utt] == 'sad':
                hss += 1
                h_s += 1
            #########################################################################
            if pre_pre_emo == 'neu' and pre_emo == 'ang' and emo_dict[utt] == 'ang':
                naa += 1
                n_a += 1
            if pre_pre_emo == 'neu' and pre_emo == 'ang' and emo_dict[utt] == 'hap':
                nah += 1
                n_a += 1
            if pre_pre_emo == 'neu' and pre_emo == 'ang' and emo_dict[utt] == 'neu':
                nan += 1
                n_a += 1
            if pre_pre_emo == 'neu' and pre_emo == 'ang' and emo_dict[utt] == 'sad':
                nas += 1
                n_a += 1

            if pre_pre_emo == 'neu' and pre_emo == 'hap' and emo_dict[utt] == 'ang':
                nha += 1
                n_h += 1
            if pre_pre_emo == 'neu' and pre_emo == 'hap' and emo_dict[utt] == 'hap':
                nhh += 1
                n_h += 1
            if pre_pre_emo == 'neu' and pre_emo == 'hap' and emo_dict[utt] == 'neu':
                nhn += 1
                n_h += 1
            if pre_pre_emo == 'neu' and pre_emo == 'hap' and emo_dict[utt] == 'sad':
                nhs += 1
                n_h += 1

            if pre_pre_emo == 'neu' and pre_emo == 'neu' and emo_dict[utt] == 'ang':
                nna += 1
                n_n += 1
            if pre_pre_emo == 'neu' and pre_emo == 'neu' and emo_dict[utt] == 'hap':
                nnh += 1
                n_n += 1
            if pre_pre_emo == 'neu' and pre_emo == 'neu' and emo_dict[utt] == 'neu':
                nnn += 1
                n_n += 1
            if pre_pre_emo == 'neu' and pre_emo == 'neu' and emo_dict[utt] == 'sad':
                nns += 1
                n_n += 1

            if pre_pre_emo == 'neu' and pre_emo == 'sad' and emo_dict[utt] == 'ang':
                nsa += 1
                n_s += 1
            if pre_pre_emo == 'neu' and pre_emo == 'sad' and emo_dict[utt] == 'hap':
                nsh += 1
                n_s += 1
            if pre_pre_emo == 'neu' and pre_emo == 'sad' and emo_dict[utt] == 'neu':
                nsn += 1
                n_s += 1
            if pre_pre_emo == 'neu' and pre_emo == 'sad' and emo_dict[utt] == 'sad':
                nss += 1          
                n_s += 1
            #########################################################################
            if pre_pre_emo == 'sad' and pre_emo == 'ang' and emo_dict[utt] == 'ang':
                saa += 1
                s_a += 1
            if pre_pre_emo == 'sad' and pre_emo == 'ang' and emo_dict[utt] == 'hap':
                sah += 1
                s_a += 1
            if pre_pre_emo == 'sad' and pre_emo == 'ang' and emo_dict[utt] == 'neu':
                san += 1
                s_a += 1
            if pre_pre_emo == 'sad' and pre_emo == 'ang' and emo_dict[utt] == 'sad':
                sas += 1
                s_a += 1

            if pre_pre_emo == 'sad' and pre_emo == 'hap' and emo_dict[utt] == 'ang':
                sha += 1
                s_h += 1
            if pre_pre_emo == 'sad' and pre_emo == 'hap' and emo_dict[utt] == 'hap':
                shh += 1
                s_h += 1
            if pre_pre_emo == 'sad' and pre_emo == 'hap' and emo_dict[utt] == 'neu':
                shn += 1
                s_h += 1
            if pre_pre_emo == 'sad' and pre_emo == 'hap' and emo_dict[utt] == 'sad':
                shs += 1
                s_h += 1

            if pre_pre_emo == 'sad' and pre_emo == 'neu' and emo_dict[utt] == 'ang':
                sna += 1
                s_n += 1
            if pre_pre_emo == 'sad' and pre_emo == 'neu' and emo_dict[utt] == 'hap':
                snh += 1
                s_n += 1
            if pre_pre_emo == 'sad' and pre_emo == 'neu' and emo_dict[utt] == 'neu':
                snn += 1
                s_n += 1
            if pre_pre_emo == 'sad' and pre_emo == 'neu' and emo_dict[utt] == 'sad':
                sns += 1
                s_n += 1

            if pre_pre_emo == 'sad' and pre_emo == 'sad' and emo_dict[utt] == 'ang':
                ssa += 1
                s_s += 1
            if pre_pre_emo == 'sad' and pre_emo == 'sad' and emo_dict[utt] == 'hap':
                ssh += 1
                s_s += 1
            if pre_pre_emo == 'sad' and pre_emo == 'sad' and emo_dict[utt] == 'neu':
                ssn += 1
                s_s += 1
            if pre_pre_emo == 'sad' and pre_emo == 'sad' and emo_dict[utt] == 'sad':
                sss += 1
                s_s += 1
   
            pre_dialog_id = dialog_id
            pre_pre_emo = pre_emo
            pre_emo = emo_dict[utt]

    if add_one_smooth_or_not == 1:
        a_a += 3
        a_h += 4
        a_n += 4
        a_s += 4

        h_a += 4
        h_h += 4
        h_n += 4
        h_s += 4

        n_a += 4
        n_h += 4
        n_n += 4
        n_s += 4

        s_a += 4
        s_h += 4
        s_n += 4
        s_s += 4

        aaa += 1
        aah += 1
        aan += 1
        aas += 1
        aha += 1
        ahh += 1
        ahn += 1
        ahs += 1
        ana += 1
        anh += 1
        ann += 1
        ans += 1
        asa += 1
        ash += 1
        asn += 1
        ass += 1

        haa += 1
        hah += 1
        han += 1
        has += 1
        hha += 1
        hhh += 1
        hhn += 1
        hhs += 1
        hna += 1
        hnh += 1
        hnn += 1
        hns += 1
        hsa += 1
        hsh += 1
        hsn += 1
        hss += 1

        naa += 1
        nah += 1
        nan += 1
        nas += 1
        nha += 1
        nhh += 1
        nhn += 1
        nhs += 1
        nna += 1
        nnh += 1
        nnn += 1
        nns += 1
        nsa += 1
        nsh += 1
        nsn += 1
        nss += 1

        saa += 1
        sah += 1
        san += 1
        sas += 1
        sha += 1
        shh += 1
        shn += 1
        shs += 1
        sna += 1
        snh += 1
        snn += 1
        sns += 1
        ssa += 1
        ssh += 1
        ssn += 1
        sss += 1
    print( (aaa+aah+aan+aas)/a_a )
    print( (aha+ahh+ahn+ahs)/a_h )
    print( (ana+anh+ann+ans)/a_n )
    print( (asa+ash+asn+ass)/a_s )

    print( (haa+hah+han+has)/h_a )
    print( (hha+hhh+hhn+hhs)/h_h )
    print( (hna+hnh+hnn+hns)/h_n )
    print( (hsa+hsh+hsn+hss)/h_s )

    print( (naa+nah+nan+nas)/n_a )
    print( (nha+nhh+nhn+nhs)/n_h )
    print( (nna+nnh+nnn+nns)/n_n )
    print( (nsa+nsh+nsn+nss)/n_s )

    print( (saa+sah+san+sas)/s_a )
    print( (sha+shh+shn+shs)/s_h )
    print( (sna+snh+snn+sns)/s_n )
    print( (ssa+ssh+ssn+sss)/s_s )
    print('=============================================')
    return {'aaa':aaa/a_a, 'aah':aah/a_a, 'aan':aan/a_a, 'aas':aas/a_a, \
            'aha':aha/a_h, 'ahh':ahh/a_h, 'ahn':ahn/a_h, 'ahs':ahs/a_h, \
            'ana':ana/a_n, 'anh':anh/a_n, 'ann':ann/a_n, 'ans':ans/a_n, \
            'asa':asa/a_s, 'ash':ash/a_s, 'asn':asn/a_s, 'ass':ass/a_s, \
            'haa':haa/h_a, 'hah':hah/h_a, 'han':han/h_a, 'has':has/h_a, \
            'hha':hha/h_h, 'hhh':hhh/h_h, 'hhn':hhn/h_h, 'hhs':hhs/h_h, \
            'hna':hna/h_n, 'hnh':hnh/h_n, 'hnn':hnn/h_n, 'hns':hns/h_n, \
            'hsa':hsa/h_s, 'hsh':hsh/h_s, 'hsn':hsn/h_s, 'hss':hss/h_s, \
            'naa':naa/n_a, 'nah':nah/n_a, 'nan':nan/n_a, 'nas':nas/n_a, \
            'nha':nha/n_h, 'nhh':nhh/n_h, 'nhn':nhn/n_h, 'nhs':nhs/n_h, \
            'nna':nna/n_n, 'nnh':nnh/n_n, 'nnn':nnn/n_n, 'nns':nns/n_n, \
            'nsa':nsa/n_s, 'nsh':nsh/n_s, 'nsn':nsn/n_s, 'nss':nss/n_s, \
            'saa':saa/s_a, 'sah':sah/s_a, 'san':san/s_a, 'sas':sas/s_a, \
            'sha':sha/s_h, 'shh':shh/s_h, 'shn':shn/s_h, 'shs':shs/s_h, \
            'sna':sna/s_n, 'snh':snh/s_n, 'snn':snn/s_n, 'sns':sns/s_n, \
            'ssa':ssa/s_s, 'ssh':ssh/s_s, 'ssn':ssn/s_s, 'sss':sss/s_s  }

def get_val_emo_trans_prob(emo_dict, dialogs, softmax_or_not, Bi_or_Tri):
    """Get emo_trans_prob estimated from training sessions."""

    session = ['Ses01', 'Ses02', 'Ses03', 'Ses04', 'Ses05']
    emo_trans_prob_dict = {}
    for i in range(len(session)):
        val = session[i]
        train_sessions = session[:i] + session[i+1:]
        if Bi_or_Tri == 2:
            if softmax_or_not == 1:
                emo_trans_prob_com = emo_trans_prob_BI_need_softmax(emo_dict, dialogs, val)
                emo_trans_prob_dict[val] = emo_trans_prob_com
            elif softmax_or_not == 0:
                emo_trans_prob_com = emo_trans_prob_BI_without_softmax(emo_dict, dialogs, val)
                emo_trans_prob_dict[val] = emo_trans_prob_com
        elif Bi_or_Tri == 3:
            if softmax_or_not == 1:
                emo_trans_prob_com = emo_trans_prob_TRI_need_softmax(emo_dict, dialogs, 0, val)
                emo_trans_prob_dict[val] = emo_trans_prob_com
            elif softmax_or_not == 0:
                emo_trans_prob_com = emo_trans_prob_TRI_without_softmax(emo_dict, dialogs, 1, val)
                emo_trans_prob_dict[val] = emo_trans_prob_com
    return emo_trans_prob_dict

def find_last_idx(trace_speakers, speaker):
  """Find the index of speaker's last utterance."""
  for i in range(len(trace_speakers)):
    if trace_speakers[len(trace_speakers) - (i+1)] == speaker:
        return len(trace_speakers) - (i+1)

def find_second_to_last_idx(trace_speakers, speaker):
  """Find the index of speaker's second_to_last utterance."""
  cnt = 1
  for i in range(len(trace_speakers)):
    if trace_speakers[len(trace_speakers) - (i+1)] == speaker and cnt == 2:
        return len(trace_speakers) - (i+1)
    if trace_speakers[len(trace_speakers) - (i+1)] == speaker:
        cnt += 1

def cross_entropy(targets, predictions, epsilon=1e-12):
  """Computes cross entropy between targets (one-hot) and predictions. 
  Args: 
    targets: (1, num_state) ndarray.   
    predictions: (1, num_state) ndarray.

  Returns: 
    cross entropy loss.
  """
  targets = np.array(targets)
  predictions = np.array(predictions)
  ce = -np.sum(targets*predictions)
  return ce

def convert_to_index(emotion):
  """convert emotion to index """
  map_emo = {'ang':0, 'hap':1, 'neu':2, 'sad':3}
  if emotion in map_emo.keys():
    return map_emo[emotion]
  else:
    return -1

def evaluate(trace, label):
  # Only evaluate utterances labeled in defined 4 emotion states
  label, trace = np.array(label), np.array(trace)
  index = [label != -1]
  label, trace = label[index], trace[index]

  return recall_score(label, trace, average='macro'), accuracy_score(label, trace), confusion_matrix(label, trace)


if __name__ == '__main__':
    dialog = {'Ses05M_script03_2_M': ['Ses05M_script03_2_M042', 'Ses05M_script03_2_M043', 
                'Ses05M_script03_2_M044', 'Ses05M_script03_2_M045']}
    emo = {'Ses05M_script03_2_M042': 'ang', 'Ses05M_script03_2_M043': 'ang', 
                'Ses05M_script03_2_M044': 'ang', 'Ses05M_script03_2_M045': 'ang'}

    spk_dialog = split_dialog(dialog)
    bias, total_transit = transition_bias(spk_dialog, emo)
    crp_alpha = 1
    print('Transition bias: {} , Total transition: {}'.format(bias, total_transit))

