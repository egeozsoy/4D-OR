'''
Uses the predicted scene graphs to generate a surgery report.
'''
import json
from pathlib import Path
from typing import List, Optional

from helpers.inm_helpers.configurations import TAKE_SPLIT


def augment_scene_graphs_with_roles(rels, key, role_predictions):
    key = key.rsplit('_', 1)[0]
    if key in role_predictions:
        role_predictions = role_predictions[key]
        for rel_idx, (sub_name, rel_name, obj_name) in enumerate(rels):
            sub_name = role_predictions.get(sub_name, sub_name)
            obj_name = role_predictions.get(obj_name, obj_name)
            rels[rel_idx] = (sub_name, rel_name, obj_name)

        return rels

    else:
        return rels


def q(relationships: List, s: Optional[List] = None, r: Optional[List] = None, o: Optional[List] = None):
    '''
    :param relationships: Human Readable Relationships
    :param s: subjects
    :param r: relations
    :param o: objects
    '''
    # Check if relationships fullfills a condition
    for sub, rel, obj in relationships:
        if s is not None and sub not in s:
            continue
        if r is not None and rel not in r:
            continue
        if o is not None and obj not in o:
            continue
        return True
    return False


def cond(counter, window, threshold):
    # Check if there is a continuous sequence of length window, with at least threshold elements
    for center_elem in counter:
        all_elems_in_window = [elem for elem in counter if abs(elem - center_elem) <= window // 2]
        if len(all_elems_in_window) >= threshold:
            return all_elems_in_window[0], True  # return the start element, and True
    return None, False


def get_first_last_sawing(sgs, role_predictions):
    all_sawing_starts = []
    sawing_counter = []
    for key, pred_sg in sgs:
        pred_sg = augment_scene_graphs_with_roles(pred_sg, key, role_predictions)
        scan_idx = int(key.split('_')[1])
        if q(pred_sg, s=['head_surgeon'], r=['Sawing'], o=['Patient']):
            sawing_counter.append(scan_idx)
            start, flag = cond(sawing_counter, window=10, threshold=3)
            if flag:
                all_sawing_starts.append(start)
                sawing_counter = []
    first_sawing, last_sawing = all_sawing_starts[0], all_sawing_starts[-1] + 5
    return first_sawing, last_sawing


if __name__ == '__main__':
    predicted_sgs_path = Path('scan_relations_training_no_gt_images_test_scans.json')  # TODO adjust as necessary
    with open('surgery_phase_recognition/rule_based_role_predictions_with_GT_False_True.json') as f:  # TODO adjust as necessary
        role_predictions = json.load(f)

    with open(predicted_sgs_path, 'r') as f:
        predicted_sgs = json.load(f)

    for split_name in ['train', 'val', 'test']:
        for take_idx in TAKE_SPLIT[split_name]:
            phase_start = [('sterile', 0)]
            phase_end = []
            current_phase = 'sterile'
            memory = {'patient_in_counter': [], 'patient_prep_counter': [], 'cleaning_counter': [], 'cleaning_done': False, 'incision_counter': [],
                      'drilling_counter': [], 'drilling_done': False, 'hammering_counter': [], 'cementing_counter': [],
                      'cementing_done': False, 'suturing_counter': [], 'patient_out_counter': [], 'cleanup_counter': []}
            sgs = sorted({key: pred_sg for key, pred_sg in predicted_sgs.items() if int(key.split('_')[0]) == take_idx}.items())

            if len(sgs) == 0:
                continue

            first_sawing, last_sawing = get_first_last_sawing(sgs, role_predictions)

            for key, pred_sg in sgs:
                pred_sg = augment_scene_graphs_with_roles(pred_sg, key, role_predictions)
                scan_idx = int(key.split('_')[1])

                # sterile -> roll_in
                if current_phase == 'sterile' and (q(pred_sg, s=['Patient']) or q(pred_sg, o=['Patient'])) and q(pred_sg, r=['Operating'],
                                                                                                                 o=['operating_table']):
                    memory['patient_in_counter'].append(scan_idx)
                    start, flag = cond(memory['patient_in_counter'], window=5, threshold=3)
                    if flag:
                        phase_end.append((current_phase, start - 1))
                        current_phase = 'roll_in'
                        phase_start.append((current_phase, start))

                # roll_in -> patient_prep
                elif current_phase == 'roll_in' and q(pred_sg, s=['head_surgeon'], r=['Preparing']) and q(pred_sg, s=['assistant_surgeon'], r=['Preparing']):
                    memory['patient_prep_counter'].append(scan_idx)
                    start, flag = cond(memory['patient_prep_counter'], window=10, threshold=3)
                    if flag:
                        phase_end.append((current_phase, start - 1))
                        current_phase = 'patient_prep'
                        phase_start.append((current_phase, start))

                # patient_prep -> knee_prep with a sanity check: Cleaning Done
                elif current_phase == 'patient_prep':
                    if not memory['cleaning_done']:
                        if q(pred_sg, s=['head_surgeon', 'assistant_surgeon'], r=['Cleaning'], o=['Patient']):
                            memory['cleaning_counter'].append(scan_idx)
                            start, flag = cond(memory['cleaning_counter'], window=10, threshold=3)
                            if flag:
                                memory['cleaning_done'] = True
                    else:
                        # patient_prep -> knee_prep
                        if q(pred_sg, s=['head_surgeon', 'assistant_surgeon'], r=['Cutting']):
                            memory['incision_counter'].append(scan_idx)
                            start, flag = cond(memory['incision_counter'], window=10, threshold=3)
                            if flag:
                                phase_end.append((current_phase, start - 1))
                                current_phase = 'knee_prep'
                                phase_start.append((current_phase, start))

                # knee_prep -> knee_insert with a sanity check: Sawing and Drilling Done
                elif current_phase == 'knee_prep':
                    if scan_idx > last_sawing and q(pred_sg, s=['head_surgeon'], r=['Hammering'], o=['Patient']):
                        memory['hammering_counter'].append(scan_idx)
                        start, flag = cond(memory['hammering_counter'], window=5, threshold=3)
                        if flag:
                            phase_end.append((current_phase, start - 1))
                            current_phase = 'knee_insert'
                            phase_start.append((current_phase, start))

                # knee_insert -> surgery_conclusion with a sanity check: Cementing Done
                elif current_phase == 'knee_insert':
                    if not memory['cementing_done']:
                        if q(pred_sg, s=['head_surgeon', 'assistant_surgeon'], r=['Cementing'], o=['Patient']):
                            memory['cementing_counter'].append(scan_idx)
                            start, flag = cond(memory['cementing_counter'], window=10, threshold=3)
                            if flag:
                                memory['cementing_done'] = True
                    else:
                        # knee_insert -> surgery_conclusion
                        if q(pred_sg, s=['head_surgeon', 'assistant_surgeon'], r=['Suturing'], o=['Patient']):
                            memory['suturing_counter'].append(scan_idx)
                            start, flag = cond(memory['suturing_counter'], window=10, threshold=2)
                            if flag:
                                phase_end.append((current_phase, start - 1))
                                current_phase = 'surgery_conclusion'
                                phase_start.append((current_phase, start))

                # surgery_conclusion -> roll_out
                elif current_phase == 'surgery_conclusion' and (q(pred_sg, s=['Patient']) or q(pred_sg, o=['Patient'])) and q(pred_sg, r=['Operating'],
                                                                                                                              o=['operating_table']):
                    memory['patient_out_counter'].append(scan_idx)
                    start, flag = cond(memory['patient_out_counter'], window=10, threshold=8)
                    if flag:
                        phase_end.append((current_phase, start - 1))
                        current_phase = 'roll_out'
                        phase_start.append((current_phase, start))

                # roll_out -> cleanup
                elif current_phase == 'roll_out' and not q(pred_sg, o=['Patient']) and q(pred_sg, s=['circulating_nurse', 'anaesthetist']):
                    memory['cleanup_counter'].append(scan_idx)
                    start, flag = cond(memory['cleanup_counter'], window=10, threshold=3)
                    if flag:
                        phase_end.append((current_phase, start - 1))
                        current_phase = 'cleanup'
                        phase_start.append((current_phase, start))
                        phase_end.append((current_phase, int(sgs[-1][0].split('_')[1])))  # last scan

            # Merge phase_start and phase_end into phase_to_frames
            phase_to_frames = {}
            for (phase_s, start), (phase_e, end) in zip(phase_start, phase_end):
                assert phase_s == phase_e
                phase_to_frames[phase_s] = (start, end)

            with Path(f'surgery_phase_recognition/phases_to_frames/{predicted_sgs_path.stem}_phase_to_frames_{take_idx}.json').open('w') as f:
                json.dump(phase_to_frames, f)
