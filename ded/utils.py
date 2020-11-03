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

def emo_trans_prob_need_softmax(emo_dict, dialogs, val=None):
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
                total_transit += 1
                continue

            if pre_dialog_id != dialog_id: #new dialog
                total_transit -= 1
                pre_emo = ''

            if pre_emo == 'ang' and emo_dict[utt] == 'ang':
                ang2ang += 1
            if pre_emo == 'ang' and emo_dict[utt] == 'hap':
                ang2hap += 1
            if pre_emo == 'ang' and emo_dict[utt] == 'neu':
                ang2neu += 1
            if pre_emo == 'ang' and emo_dict[utt] == 'sad':
                ang2sad += 1

            if pre_emo == 'hap' and emo_dict[utt] == 'ang':
                hap2ang += 1
            if pre_emo == 'hap' and emo_dict[utt] == 'hap':
                hap2hap += 1
            if pre_emo == 'hap' and emo_dict[utt] == 'neu':
                hap2neu += 1
            if pre_emo == 'hap' and emo_dict[utt] == 'sad':
                hap2sad += 1

            if pre_emo == 'neu' and emo_dict[utt] == 'ang':
                neu2ang += 1
            if pre_emo == 'neu' and emo_dict[utt] == 'hap':
                neu2hap += 1
            if pre_emo == 'neu' and emo_dict[utt] == 'neu':
                neu2neu += 1
            if pre_emo == 'neu' and emo_dict[utt] == 'sad':
                neu2sad += 1

            if pre_emo == 'sad' and emo_dict[utt] == 'ang':
                sad2ang += 1
            if pre_emo == 'sad' and emo_dict[utt] == 'hap':
                sad2hap += 1
            if pre_emo == 'sad' and emo_dict[utt] == 'neu':
                sad2neu += 1
            if pre_emo == 'sad' and emo_dict[utt] == 'sad':
                sad2sad += 1
            
            pre_dialog_id = dialog_id
            pre_emo = emo_dict[utt]
            total_transit += 1
    if last_dialog_last_utt_emo == 'ang' or last_dialog_last_utt_emo == 'hap' or last_dialog_last_utt_emo == 'neu' or last_dialog_last_utt_emo == 'sad':
        total_transit -= 1
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

def emo_trans_prob_without_softmax(emo_dict, dialogs, val=None):
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
                if pre_emo == 'ang':
                    a2 += 1
                elif pre_emo == 'hap':
                    h2 += 1
                elif pre_emo == 'neu':
                    n2 += 1
                else:
                    s2 += 1
                continue

            if pre_dialog_id != dialog_id: #new dialog
                if pre_emo == 'ang':
                    a2 -= 1
                elif pre_emo == 'hap':
                    h2 -= 1
                elif pre_emo == 'neu':
                    n2 -= 1
                else:
                    s2 -= 1
                pre_emo = ''

            if pre_emo == 'ang' and emo_dict[utt] == 'ang':
                ang2ang += 1
            if pre_emo == 'ang' and emo_dict[utt] == 'hap':
                ang2hap += 1
            if pre_emo == 'ang' and emo_dict[utt] == 'neu':
                ang2neu += 1
            if pre_emo == 'ang' and emo_dict[utt] == 'sad':
                ang2sad += 1

            if pre_emo == 'hap' and emo_dict[utt] == 'ang':
                hap2ang += 1
            if pre_emo == 'hap' and emo_dict[utt] == 'hap':
                hap2hap += 1
            if pre_emo == 'hap' and emo_dict[utt] == 'neu':
                hap2neu += 1
            if pre_emo == 'hap' and emo_dict[utt] == 'sad':
                hap2sad += 1

            if pre_emo == 'neu' and emo_dict[utt] == 'ang':
                neu2ang += 1
            if pre_emo == 'neu' and emo_dict[utt] == 'hap':
                neu2hap += 1
            if pre_emo == 'neu' and emo_dict[utt] == 'neu':
                neu2neu += 1
            if pre_emo == 'neu' and emo_dict[utt] == 'sad':
                neu2sad += 1

            if pre_emo == 'sad' and emo_dict[utt] == 'ang':
                sad2ang += 1
            if pre_emo == 'sad' and emo_dict[utt] == 'hap':
                sad2hap += 1
            if pre_emo == 'sad' and emo_dict[utt] == 'neu':
                sad2neu += 1
            if pre_emo == 'sad' and emo_dict[utt] == 'sad':
                sad2sad += 1
            
            pre_dialog_id = dialog_id
            pre_emo = emo_dict[utt]
            if pre_emo == 'ang':
                a2 += 1
            elif pre_emo == 'hap':
                h2 += 1
            elif pre_emo == 'neu':
                n2 += 1
            else:
                s2 += 1
    
    if last_dialog_last_utt_emo == 'ang':
            a2 -= 1
    elif last_dialog_last_utt_emo == 'hap':
        h2 -= 1
    elif last_dialog_last_utt_emo == 'neu':
        n2 -= 1
    elif last_dialog_last_utt_emo == 'sad':
        s2 -= 1
    print(ang2ang/a2+ang2hap/a2+ang2neu/a2+ang2sad/a2)
    print(hap2ang/h2+hap2hap/h2+hap2neu/h2+hap2sad/h2)
    print(neu2ang/n2+neu2hap/n2+neu2neu/n2+neu2sad/n2)
    print(sad2ang/s2+sad2hap/s2+sad2neu/s2+sad2sad/s2)
    print('=============================================')
    return {'a2a':ang2ang/a2, 'a2h':ang2hap/a2, 'a2n':ang2neu/a2, 'a2s':ang2sad/a2, \
                    'h2a':hap2ang/h2, 'h2h':hap2hap/h2, 'h2n':hap2neu/h2, 'h2s':hap2sad/h2, \
                    'n2a':neu2ang/n2, 'n2h':neu2hap/n2, 'n2n':neu2neu/n2, 'n2s':neu2sad/n2, \
                    's2a':sad2ang/s2, 's2h':sad2hap/s2, 's2n':sad2neu/s2, 's2s':sad2sad/s2}

def get_val_emo_trans_prob(emo_dict, dialogs, softmax_or_not):
    """Get emo_trans_prob estimated from training sessions."""

    session = ['Ses01', 'Ses02', 'Ses03', 'Ses04', 'Ses05']
    emo_trans_prob_dict = {}
    for i in range(len(session)):
        val = session[i]
        train_sessions = session[:i] + session[i+1:]
        if softmax_or_not == 1:
            emo_trans_prob_com = emo_trans_prob_need_softmax(emo_dict, dialogs, val)
            emo_trans_prob_dict[val] = emo_trans_prob_com
        elif softmax_or_not == 0:
            emo_trans_prob_com = emo_trans_prob_without_softmax(emo_dict, dialogs, val)
            emo_trans_prob_dict[val] = emo_trans_prob_com
    return emo_trans_prob_dict

def find_last_idx(trace_speakers, speaker):
  """Find the index of speaker's last utterance."""
  for i in range(len(trace_speakers)):
    if trace_speakers[len(trace_speakers) - (i+1)] == speaker:
        return len(trace_speakers) - (i+1)

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

