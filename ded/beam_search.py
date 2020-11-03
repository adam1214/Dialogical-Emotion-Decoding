import numpy as np
from ded import utils

#TODO change spk_sequence to indexes, record last state of the speaker in Beam.

class BeamState(object):

    def __init__(self, spk_sequence, emo_sequence, log_prob, block_counts):
        """hypothesis
        Args:
            spk_sequence: List, sequence of speaker indexes in dialogical order.
            emo_sequence: List, sequence of emotion state indexes in dialogical order.
            block_counts: List, initially 0 block for each emotion state, note that states are bounded.
        """
        self.spk_sequence = spk_sequence
        self.emo_sequence = emo_sequence
        self.log_prob = log_prob
        self.block_counts = block_counts

    def update(self, spk, state, log_prob):
        """Return new beam state based on last decoding results."""
        return  BeamState(
                          self.spk_sequence+[spk],
                          self.emo_sequence+[state],
                          self.log_prob+log_prob,
                          self.block_counts)
   
    def copy_beam(self):
        """Return copied beam state."""
        return  BeamState(
                          self.spk_sequence.copy(),
                          self.emo_sequence.copy(),
                          self.log_prob,
                          self.block_counts.copy())

class BeamSearch(object):

  def __init__(self, bias, emo_trans_prob_dict, crp_alpha, num_state, beam_size, test_iteration, emo, logits):
    # Define parameters
    self.transition_bias = bias
    self.emo_trans_prob_dict = emo_trans_prob_dict
    self.crp_alpha = crp_alpha
    self.num_state = num_state
    self.beam_size = beam_size
    self.test_iteration = test_iteration
    self.emo = emo
    self.all_logits = logits

  def decode(self, dialog):
    """
    Args:
        dialog: List, a list of utterances id in dialogical order.

        For example:
        ```
        dialog = 
        ['Ses01M_XX_M000', 'Ses01M_XX_F000', 'Ses01M_XX_F002', ...]
        ```
    Returns:
        predicted_sequence: Predicted emotion state sequence. An array of integers.
    """
    test_sequence_length = len(dialog)
    test_sequence = np.tile(dialog, self.test_iteration)

    # decoding steps
    dec_step = len(test_sequence)
    t = 0
    beam_set = [BeamState([], [], 0, [0, 0, 0, 0]) for _ in range(self.beam_size)]
    while t < dec_step:
      utt_id = test_sequence[t]
      beam_set_bank = []

      num_beam = len(beam_set) if t > 0 else 1

      # collect nodes
      for i in range(num_beam):
        for j in range(self.num_state):  # => search each node
          updated_beam = self._update_beam(beam_set[i].copy_beam(), utt_id, j)
          beam_set_bank.append(updated_beam)

      # sort by log prob
      beam_set = self._select_best_k(beam_set_bank)
      t += 1

    # Return the best beam state.
    return beam_set[0].emo_sequence[-len(dialog):]

  def _get_logits(self, utt_id):
    return self.all_logits[utt_id]

  def _update_beam(self, beam_state, utt_id, state):
    """Calculate log probability for shift and assignment process based on current beam state."""
    loss = 0
    speaker = utt_id[-4]
    # Convert the state into one-hot vector
    label = np.zeros(self.num_state)
    label[state] = 1

    # Get original ce loss
    logit = np.reshape(self.all_logits[utt_id], [4,])
    loss = utils.cross_entropy(label, logit)
    '''
    # RESCORING:

    # An existing state
    if state in np.unique(beam_state.emo_sequence) and speaker in np.unique(beam_state.spk_sequence): 
      # Find last state
      last_idx = utils.find_last_idx(beam_state.spk_sequence, speaker)
      last_state = beam_state.emo_sequence[last_idx]
      # No shift
      if state == last_state:
        loss -= np.log(1 - self.transition_bias)
      # Shift
      else:
        loss -= np.log(self.transition_bias) + \
                np.log(beam_state.block_counts[state]) - \
                np.log(sum(beam_state.block_counts) + self.crp_alpha)
        beam_state.block_counts[state] += 1

    # A new state
    else: 
      loss -= np.log(self.transition_bias) + \
              np.log(self.crp_alpha) - \
              np.log(sum(beam_state.block_counts) + \
              self.crp_alpha)          
      beam_state.block_counts[state] += 1
    '''
    # Find last state of this speaker
    last_idx = utils.find_last_idx(beam_state.spk_sequence, speaker)
    if last_idx == None: #this is speaker first utterance
      loss -= np.log(1)
    else:
      #{ang, hap, neu, sad}
      last_state = beam_state.emo_sequence[last_idx]
      if last_state == 0 and state == 0:
        loss -= np.log(self.emo_trans_prob_dict['a2a'])
      elif last_state == 0 and state == 1:
        loss -= np.log(self.emo_trans_prob_dict['a2h'])
      elif last_state == 0 and state == 2:
        loss -= np.log(self.emo_trans_prob_dict['a2n'])
      elif last_state == 0 and state == 3:
        loss -= np.log(self.emo_trans_prob_dict['a2s'])

      elif last_state == 1 and state == 0:
        loss -= np.log(self.emo_trans_prob_dict['h2a'])
      elif last_state == 1 and state == 1:
        loss -= np.log(self.emo_trans_prob_dict['h2h'])
      elif last_state == 1 and state == 2:
        loss -= np.log(self.emo_trans_prob_dict['h2n'])
      elif last_state == 1 and state == 3:
        loss -= np.log(self.emo_trans_prob_dict['h2s'])

      elif last_state == 2 and state == 0:
        loss -= np.log(self.emo_trans_prob_dict['n2a'])
      elif last_state == 2 and state == 1:
        loss -= np.log(self.emo_trans_prob_dict['n2h'])
      elif last_state == 2 and state == 2:
        loss -= np.log(self.emo_trans_prob_dict['n2n'])
      elif last_state == 2 and state == 3:
        loss -= np.log(self.emo_trans_prob_dict['n2s'])

      elif last_state == 3 and state == 0:
        loss -= np.log(self.emo_trans_prob_dict['s2a'])
      elif last_state == 3 and state == 1:
        loss -= np.log(self.emo_trans_prob_dict['s2h'])
      elif last_state == 3 and state == 2:
        loss -= np.log(self.emo_trans_prob_dict['s2n'])
      elif last_state == 3 and state == 3:
        loss -= np.log(self.emo_trans_prob_dict['s2s'])
    '''
    # Find last state of current dialog
    if len(beam_state.emo_sequence) == 0: #first utterance of this dialog
      loss -= np.log(1)
    else:
      #{ang, hap, neu, sad}
      last_state = beam_state.emo_sequence[len(beam_state.emo_sequence)-1]
      if last_state == 0 and state == 0:
        loss -= np.log(self.emo_trans_prob_dict['a2a'])
      elif last_state == 0 and state == 1:
        loss -= np.log(self.emo_trans_prob_dict['a2h'])
      elif last_state == 0 and state == 2:
        loss -= np.log(self.emo_trans_prob_dict['a2n'])
      elif last_state == 0 and state == 3:
        loss -= np.log(self.emo_trans_prob_dict['a2s'])

      elif last_state == 1 and state == 0:
        loss -= np.log(self.emo_trans_prob_dict['h2a'])
      elif last_state == 1 and state == 1:
        loss -= np.log(self.emo_trans_prob_dict['h2h'])
      elif last_state == 1 and state == 2:
        loss -= np.log(self.emo_trans_prob_dict['h2n'])
      elif last_state == 1 and state == 3:
        loss -= np.log(self.emo_trans_prob_dict['h2s'])

      elif last_state == 2 and state == 0:
        loss -= np.log(self.emo_trans_prob_dict['n2a'])
      elif last_state == 2 and state == 1:
        loss -= np.log(self.emo_trans_prob_dict['n2h'])
      elif last_state == 2 and state == 2:
        loss -= np.log(self.emo_trans_prob_dict['n2n'])
      elif last_state == 2 and state == 3:
        loss -= np.log(self.emo_trans_prob_dict['n2s'])

      elif last_state == 3 and state == 0:
        loss -= np.log(self.emo_trans_prob_dict['s2a'])
      elif last_state == 3 and state == 1:
        loss -= np.log(self.emo_trans_prob_dict['s2h'])
      elif last_state == 3 and state == 2:
        loss -= np.log(self.emo_trans_prob_dict['s2n'])
      elif last_state == 3 and state == 3:
        loss -= np.log(self.emo_trans_prob_dict['s2s'])
    '''
    new_beam_state = beam_state.update(speaker, state, loss)
    return new_beam_state


  def _select_best_k(self, beam_set):
    """select top k BeamState
    Args:
        beam_set: list, beam set.
    Returns:
        beam_set: list, sorted beam_set based on log_prob in acending order.
    """
    log_prob = [b.log_prob for b in beam_set]

    idx = np.argsort(log_prob)[:self.beam_size]

    return [beam_set[i] for i in idx]


