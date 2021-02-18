"""Import common classes."""
# flake8: noqa
from robustnessgym.logging.utils import (
    initialize_logging,
    set_logging_level,
    set_logging_level_for_imports,
)

initialize_logging()

from robustnessgym.cachedops.allen.allen_predictor import AllenPredictor
from robustnessgym.cachedops.allen.constituency_parser import AllenConstituencyParser
from robustnessgym.cachedops.allen.dependency_parser import AllenDependencyParser
from robustnessgym.cachedops.allen.semantic_role_labeler import AllenSemanticRoleLabeler
from robustnessgym.cachedops.bootleg import Bootleg
from robustnessgym.cachedops.similarity import (
    RougeMatrix,
    RougeScore,
    SentenceSimilarityMatrix,
)
from robustnessgym.cachedops.spacy import Spacy
from robustnessgym.cachedops.stanza import Stanza
from robustnessgym.cachedops.strip_text import StripText
from robustnessgym.cachedops.textblob import TextBlob
from robustnessgym.core.cachedops import (
    CachedOperation,
    SingleColumnCachedOperation,
    stow,
)
from robustnessgym.core.dataset import Dataset
from robustnessgym.core.identifier import Identifier
from robustnessgym.core.slice import Slice
from robustnessgym.core.testbench import TestBench
from robustnessgym.slicebuilders.attacks.textattack import TextAttack
from robustnessgym.slicebuilders.slicebuilder import (
    SliceBuilder,
    SliceBuilderCollection,
)
from robustnessgym.slicebuilders.subpopulations.constituency_overlap import (
    ConstituencyOverlapSubpopulation,
    ConstituencySubtreeSubpopulation,
    FuzzyConstituencySubtreeSubpopulation,
)
from robustnessgym.slicebuilders.subpopulations.entity_frequency import EntityFrequency
from robustnessgym.slicebuilders.subpopulations.hans import (
    HansAdjectives,
    HansAdjectivesCompEnt,
    HansAdjectivesCompNonEnt,
    HansAdverbs,
    HansAdvsEntailed,
    HansAdvsNonEntailed,
    HansAllPhrases,
    HansCalledObjects,
    HansConjs,
    HansConstAdv,
    HansConstQuotEntailed,
    HansEntComplementNouns,
    HansFoodWords,
    HansIntransitiveVerbs,
    HansLocationNounsA,
    HansLocationNounsB,
    HansNonEntComplementNouns,
    HansNonEntQuotVerbs,
    HansNPSVerbs,
    HansNPZVerbs,
    HansPassiveVerbs,
    HansPastParticiples,
    HansPluralNouns,
    HansPluralNPZVerbs,
    HansPrepositions,
    HansQuestionEmbeddingVerbs,
    HansQuestions,
    HansReadWroteObjects,
    HansRelations,
    HansSingularNouns,
    HansToldObjects,
    HansTransitiveVerbs,
    HansUnderstoodArgumentVerbs,
    HansWonObjects,
)
from robustnessgym.slicebuilders.subpopulations.length import LengthSubpopulation
from robustnessgym.slicebuilders.subpopulations.lexical_overlap import (
    LexicalOverlapSubpopulation,
)
from robustnessgym.slicebuilders.subpopulations.phrase import (
    AhoCorasick,
    HasAllPhrases,
    HasAnyPhrase,
    HasComparison,
    HasDefiniteArticle,
    HasIndefiniteArticle,
    HasNegation,
    HasPhrase,
    HasPosessivePreposition,
    HasQuantifier,
    HasTemporalPreposition,
)
from robustnessgym.slicebuilders.subpopulations.score import ScoreSubpopulation
from robustnessgym.slicebuilders.subpopulations.similarity import (
    Abstractiveness,
    Dispersion,
    Distillation,
    Ordering,
    Position,
    RougeMatrixScoreSubpopulation,
    RougeScoreSubpopulation,
)
from robustnessgym.slicebuilders.transformations.eda import EasyDataAugmentation
from robustnessgym.slicebuilders.transformations.fairseq import FairseqBacktranslation
from robustnessgym.slicebuilders.transformations.nlpaug import NlpAugTransformation
from robustnessgym.slicebuilders.transformations.similarity import (
    RougeMatrixSentenceTransformation,
)
from robustnessgym.tasks.task import (
    BinaryNaturalLanguageInference,
    BinarySentiment,
    ExtractiveQuestionAnswering,
    NaturalLanguageInference,
    QuestionAnswering,
    Sentiment,
    Summarization,
    Task,
    TernaryNaturalLanguageInference,
)

from .slicebuilders.attack import Attack
from .slicebuilders.curator import Curator
from .slicebuilders.subpopulation import Subpopulation, SubpopulationCollection

# from .attacks import *
# from .augmentations import *
# from .cache import *
# from .cache import (
#     CachedOperation,
#     stow
# )
# from .dataset import Dataset
# from .identifier import Identifier
# from .model import Model
# from .report import Report
# from .slice import Slice
# from .slicebuilders import *
# from .slicebuilders.attacks.textattack.textattack import TextAttack
# from .slicebuilders.slicebuilder import (
#     SliceBuilder,
# )
# from .slicebuilders.subpopulations.constituency_overlap.constituency_overlap import (
#     HasConstituencyOverlap,
#     HasConstituencySubtree,
#     HasFuzzyConstituencySubtree,
# )
# from .slicebuilders.subpopulations.length.length import HasLength
# from .slicebuilders.subpopulations.ner.entity_frequency import EntityFrequency
# from .slicebuilders.subpopulations.phrase.hans import (
#     HansAllPhrases,
#     HansSingularNouns,
#     HansPluralNouns,
#     HansTransitiveVerbs,
#     HansPassiveVerbs,
#     HansIntransitiveVerbs,
#     HansNPSVerbs,
#     HansNPZVerbs,
#     HansPluralNPZVerbs,
#     HansPrepositions,
#     HansConjs,
#     HansPastParticiples,
#     HansUnderstoodArgumentVerbs,
#     HansNonEntQuotVerbs,
#     HansQuestionEmbeddingVerbs,
#     HansCalledObjects,
#     HansToldObjects,
#     HansFoodWords,
#     HansLocationNounsA,
#     HansLocationNounsB,
#     HansWonObjects,
#     HansReadWroteObjects,
#     HansAdjectives,
#     HansAdjectivesCompNonEnt,
#     HansAdjectivesCompEnt,
#     HansAdverbs,
#     HansConstAdv,
#     HansConstQuotEntailed,
#     HansRelations,
#     HansQuestions,
#     HansNonEntComplementNouns,
#     HansEntComplementNouns,
#     HansAdvsNonEntailed,
#     HansAdvsEntailed,
# )
# from .slicebuilders.subpopulations.phrase.phrase import (
#     AhoCorasick,
#     HasPhrase,
#     HasAnyPhrase,
#     HasAllPhrases,
# )
# from .slicebuilders.subpopulations.phrase.wordlists import (
#     HasCategoryPhrase
# )
# from .storage import PicklerMixin
# from .task import (
#     Task,
#     NaturalLanguageInference,
#     BinaryNaturalLanguageInference,
#     TernaryNaturalLanguageInference,
# )
# from .testbench.testbench import TestBench
# from .tools import (
#     recmerge,
#     persistent_hash,
# )
