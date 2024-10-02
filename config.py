from mordred import Autocorrelation, InformationContent, MoeType, Constitutional
from pathlib import Path

PROJECT_PATH = Path.cwd()

#RANDOM_STATE = 42


MODEL = 'XGBClassifier.pkl'

BORUTA_FEATURES = [Autocorrelation.AATS(0, 'se'),
 Autocorrelation.AATS(1, 'se'),
 Autocorrelation.AATS(0, 'pe'),
 Autocorrelation.AATS(1, 'pe'),
 Autocorrelation.AATS(0, 'are'),
 Autocorrelation.AATS(1, 'are'),
 Autocorrelation.ATSC(0, 'c'),
 Autocorrelation.ATSC(4, 'c'),
 Autocorrelation.ATSC(5, 'd'),
 Autocorrelation.ATSC(1, 'se'),
 Autocorrelation.ATSC(1, 'are'),
 Autocorrelation.AATSC(0, 'c'),
 Autocorrelation.AATSC(1, 'c'),
 Autocorrelation.AATSC(5, 'd'),
 Autocorrelation.AATSC(0, 'se'),
 Autocorrelation.AATSC(1, 'se'),
 Autocorrelation.AATSC(0, 'pe'),
 Autocorrelation.AATSC(1, 'pe'),
 Autocorrelation.AATSC(0, 'are'),
 Autocorrelation.MATS(5, 'd'),
 Autocorrelation.MATS(1, 'se'),
 Autocorrelation.GATS(5, 'c'),
 Autocorrelation.GATS(8, 'c'),
 Autocorrelation.GATS(5, 'dv'),
 Autocorrelation.GATS(1, 'se'),
 Autocorrelation.GATS(5, 'pe'),
 Autocorrelation.GATS(5, 'are'),
 Constitutional.ConstitutionalMean('se'),
 Constitutional.ConstitutionalMean('are'),
 InformationContent.InformationContent(0),
 InformationContent.ModifiedIC(0),
 MoeType.VSA_EState(3),
 MoeType.VSA_EState(6),
 MoeType.VSA_EState(9)]