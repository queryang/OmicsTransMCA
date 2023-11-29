from OmicsTransMCA_predictor.models.MCA_Dense_GEP import MCA_OmicsDense_GEP
from OmicsTransMCA_predictor.models.MCA_Dense_GEP_CNV import MCA_OmicsDense_GEP_CNV
from OmicsTransMCA_predictor.models.MCA_Dense_GEP_CNV_MUT import MCA_OmicsDense_GEP_CNV_MUT
from OmicsTransMCA_predictor.models.MCA_Dense_GEP_MUT import MCA_OmicsDense_GEP_MUT
from OmicsTransMCA_predictor.models.MCA_GEP_CNV_MUT import MCA_GEP_CNV_MUT
from OmicsTransMCA_predictor.models.TransMCA_Dense_GEP import Conv_TransMCA_GEP
from OmicsTransMCA_predictor.models.TransMCA_Dense_GEP_CNV import Conv_TransMCA_GEP_CNV
from OmicsTransMCA_predictor.models.TransMCA_Dense_GEP_CNV_MUT import Conv_TransMCA_GEP_CNV_MUT
from OmicsTransMCA_predictor.models.TransMCA_Dense_GEP_MUT import Conv_TransMCA_GEP_MUT

# More models could follow
MODEL_FACTORY = {
    'trans_mca_dense_GEP_CNV_MUT': Conv_TransMCA_GEP_CNV_MUT,
    'trans_mca_dense_GEP_CNV': Conv_TransMCA_GEP_CNV,
    'trans_mca_dense_GEP': Conv_TransMCA_GEP,
    'trans_mca_dense_GEP_MUT': Conv_TransMCA_GEP_MUT,
    'mca_dense_GEP_CNV_MUT': MCA_OmicsDense_GEP_CNV_MUT,
    'mca_dense_GEP_CNV': MCA_OmicsDense_GEP_CNV,
    'mca_dense_GEP_MUT': MCA_OmicsDense_GEP_MUT,
    'mca_dense_GEP': MCA_OmicsDense_GEP,
    'mca_GEP_CNV_MUT': MCA_GEP_CNV_MUT,
}
