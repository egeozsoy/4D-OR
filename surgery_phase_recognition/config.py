PHASE_ORDER = ['sterile', 'roll_in', 'patient_prep', 'knee_prep', 'knee_insert', 'surgery_conclusion', 'roll_out', 'cleanup']
PHASE_LONG_NAMES = ['OR Preparation', 'Patient Roll-In', 'Patient Preparation', 'Surgery 1: Implant Placement Preparation',
                    'Surgery 2: Implant Placement', 'Surgery 3: Conclusion', 'Patient Roll-Out',
                    'OR Cleanup']

PHASE_TO_ABBRV = {
    'OR Preparation': 'sterile',
    'Patient Roll-In': 'roll_in',
    'Patient Preparation': 'patient_prep',
    'Surgery 1: Implant Placement Preparation': 'knee_prep',
    'Surgery 2: Implant Placement': 'knee_insert',
    'Surgery 3: Conclusion': 'surgery_conclusion',
    'Patient Roll-Out': 'roll_out',
    'OR Cleanup': 'cleanup',
}
