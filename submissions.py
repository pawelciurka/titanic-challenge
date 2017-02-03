import os
from datetime import datetime

import numpy as np
import pandas as pd

SUBMISSION_FILES_PATH = r'./submission_files'


def form_submission_file(passenger_id, predictions):
    results = pd.DataFrame({'PassengerId': passenger_id, 'Survived': predictions.astype(np.int32)})
    results.to_csv(os.path.join(SUBMISSION_FILES_PATH, _create_submission_file_name()), index=False)


def _create_submission_file_name():
    return ('pcsub_{:%d%m%Y_%H%M}.csv').format(datetime.now())
