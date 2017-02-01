import os

import numpy as np
import pandas as pd

SUBMISSION_FILES_PATH = r'./submission_files'


def form_submission_file(passenger_id, predictions, output_file_name):
    results = pd.DataFrame({'PassengerId': passenger_id, 'Survived': predictions.astype(np.int32)})
    results.to_csv(os.path.join(SUBMISSION_FILES_PATH, output_file_name), index=False)