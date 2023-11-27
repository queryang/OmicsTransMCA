from .MCA_Dense_GEP_CNV_MUT import Conv_NonTransMCA_OmicsDense_GEP_CNV_MUT
from .TransMCA_Dense_GEP import Conv_TransMCA_GEP
from .TransMCA_Dense_GEP_CNV import Conv_TransMCA_GEP_CNV
from .TransMCA_Dense_GEP_CNV_MUT import Conv_TransMCA_GEP_CNV_MUT
from .TransMCA_Dense_GEP_MUT import Conv_TransMCA_GEP_MUT


# More models could follow
MODEL_FACTORY = {
    'conv_trans_mca_GEP_CNV_MUT': Conv_TransMCA_GEP_CNV_MUT,
    'conv_trans_mca_GEP_CNV': Conv_TransMCA_GEP_CNV,
    'conv_trans_mca_GEP': Conv_TransMCA_GEP,
    'conv_trans_mca_GEP_MUT': Conv_TransMCA_GEP_MUT,
    'conv_nontrans_mca_GEP_CNV_MUT': Conv_NonTransMCA_OmicsDense_GEP_CNV_MUT
}
