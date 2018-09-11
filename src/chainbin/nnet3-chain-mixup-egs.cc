#include <limits>
#include <algorithm>
#include <fst/fstlib.h>
#include <fst/union.h>
#include <fst/determinize.h>
#include <fst/minimize.h>
#include <fst/rmepsilon.h>
#include <fst/topsort.h>
#include <boost/shared_ptr.hpp>
#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/beta_distribution.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/case_conv.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/classification.hpp>
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "nnet3/nnet-chain-example.h"

namespace kaldi { namespace nnet3 {

typedef Matrix<BaseFloat> KaldiMatrix;
typedef Vector<BaseFloat> KaldiVector;
typedef SparseMatrix<BaseFloat> KaldiSparMatrix;
typedef SparseVector<BaseFloat> KaldiSparVector;
typedef SubMatrix<BaseFloat> KaldiSubMatrix;
typedef SubVector<BaseFloat> KaldiSubVector;
typedef boost::shared_ptr<NnetChainExample> ExamplePtr;
typedef std::pair<std::string, ExamplePtr> ExamplePair;

class IRandomScale {
public:
    virtual float Value() = 0;
};

class RandomScaleUniform: public IRandomScale {
protected:
    typedef boost::random::uniform_real_distribution<float> real_distrib_t;
    typedef boost::shared_ptr<real_distrib_t> distrib_ptr_t;

protected:
    boost::random::mt19937& rand_gen;
    distrib_ptr_t distrib;

public:
    RandomScaleUniform(boost::random::mt19937& _rand_gen, const std::string& _params);
    float Value() override;
};

RandomScaleUniform::RandomScaleUniform(boost::random::mt19937& _rand_gen, const std::string& _params): rand_gen(_rand_gen), distrib() {
    std::vector<std::string> parts;
    boost::split(parts, _params, boost::is_any_of(":"));
    if (parts.size() != 2) {
        KALDI_ERR << "Wrong uniform distribution parameters string: \"" << _params << "\".";
    }
    if (parts.front() != "uniform") {
        KALDI_ERR << "Wrong uniform distribution parameters string: \"" << _params << "\".";
    }
    std::vector<std::string> min_max;
    boost::split(min_max, parts.back(), boost::is_any_of(","));
    if (parts.size() != 2) {
        KALDI_ERR << "Wrong uniform distribution parameters string: \"" << _params << "\".";
    }
    const auto min_value = boost::lexical_cast<float>(min_max.front());
    const auto max_value = boost::lexical_cast<float>(min_max.back());
    if (min_value > max_value) {
        KALDI_ERR << "min_value must be less or equal max_value (" << _params << ")";
    }
    if ((min_value < 0.0f) || (min_value > 1.0f)) {
        KALDI_ERR << "min_value must be in the range [0, 1] (" << _params << ")";
    }
    if ((max_value < 0.0f) || (max_value > 1.0f)) {
        KALDI_ERR << "max_value must be in the range [0, 1] (" << _params << ")";
    }
    distrib.reset(new real_distrib_t(min_value, max_value));
}

float RandomScaleUniform::Value() {
    return (*distrib)(rand_gen);
}

class RandomScaleBeta: public IRandomScale {
protected:
    typedef boost::random::beta_distribution<double> real_distrib_t;
    typedef boost::shared_ptr<real_distrib_t> distrib_ptr_t;

protected:
    boost::random::mt19937& rand_gen;
    distrib_ptr_t distrib;

public:
    RandomScaleBeta(boost::random::mt19937& _rand_gen, const std::string& _params);
    float Value() override;
};

RandomScaleBeta::RandomScaleBeta(boost::random::mt19937& _rand_gen, const std::string& _params): rand_gen(_rand_gen), distrib() {
    std::vector<std::string> parts;
    boost::split(parts, _params, boost::is_any_of(":"));
    if (parts.size() != 2) {
        KALDI_ERR << "Wrong beta distribution parameters string: \"" << _params << "\".";
    }
    if ((parts.front() != "beta") && (parts.front() != "beta2")) {
        KALDI_ERR << "Wrong beta distribution parameters string: \"" << _params << "\".";
    }
    const auto alpha = boost::lexical_cast<float>(parts.back());
    if (alpha <= 0.0f) {
        KALDI_ERR << "alpha must be a positive value (" << _params << ")";
    }
    distrib.reset(new real_distrib_t(alpha, alpha));
}

float RandomScaleBeta::Value() {
    return (float)(*distrib)(rand_gen);
}

class RandomScaleBeta2: public RandomScaleBeta {
public:
    RandomScaleBeta2(boost::random::mt19937& _rand_gen, const std::string& _params);
    float Value() override;
};

RandomScaleBeta2::RandomScaleBeta2(boost::random::mt19937& _rand_gen, const std::string& _params):
    RandomScaleBeta(_rand_gen, _params)
{}

float RandomScaleBeta2::Value() {
    const double value = (*distrib)(rand_gen);
    if (value <= 0.5) {
        return (float) value;
    } else {
        return (float)(1.0 - value);
    }
}

class RandomScale {
protected:
    typedef boost::shared_ptr<IRandomScale> rand_scale_t;

protected:
    rand_scale_t rand_scale;

public:
    RandomScale(boost::random::mt19937& _rand_gen, const std::string& _params);
    float operator()();
};

RandomScale::RandomScale(boost::random::mt19937& _rand_gen, const std::string& _params): rand_scale() {
    if (_params.find("uniform") == 0) {
        rand_scale.reset(new RandomScaleUniform(_rand_gen, _params));
    } else if (_params.find("beta2") == 0) {
        rand_scale.reset(new RandomScaleBeta2(_rand_gen, _params));
    } else if (_params.find("beta") == 0) {
        rand_scale.reset(new RandomScaleBeta(_rand_gen, _params));
    } else {
        KALDI_ERR << "Unknown random scale generator ID: \"" << _params << "\".";
    }
}

float RandomScale::operator()() {
    return rand_scale->Value();
}

template<class Type>
class TCounter {
protected:
    Type minimum;
    Type maximum;
    double summa;
    size_t count;

public:
    TCounter():
        minimum(std::numeric_limits<Type>::max()),
        maximum(std::numeric_limits<Type>::min()),
        summa(0.0), count(0)
    {}
    Type Minimum() const {return minimum;}
    Type Maximum() const {return maximum;}
    double Average() const {return summa / count;}
    size_t Count() const {return count;}
    bool Valid() const {return (count > 0);}
    void operator+=(Type _value) {
        minimum = std::min(_value, minimum);
        maximum = std::max(_value, maximum);
        summa += _value;
        ++count;
    }
};

typedef TCounter<int> IntCounter;
typedef TCounter<float> FloatCounter;

typedef fst::StdVectorFst fst_t;
typedef fst_t::StateId state_t;
typedef fst_t::Arc arc_t;
typedef arc_t::Label label_t;
typedef arc_t::Weight wght_t;
typedef fst::ArcIterator<fst_t> iter_t;

class ExampleMixer {
protected:
    typedef boost::random::mt19937 rand_gen_t;
    typedef boost::random::uniform_int_distribution<int32_t> int_distrib_t;
    typedef boost::random::uniform_int_distribution<size_t> uint_distrib_t;
    typedef boost::random::uniform_real_distribution<float> real_distrib_t;
    typedef std::vector<ExamplePair> egs_buffer_t;
    typedef unordered_map<NnetChainExample*, egs_buffer_t, NnetChainExampleStructureHasher, NnetChainExampleStructureCompare> eg_to_egs_t;

protected:
    struct MixupData {
        int32_t row_main;
        int32_t row_admx;
        int32_t label_main;
        int32_t label_admx;
        float scale_main;
        float scale_admx;
        MixupData(): row_main(0), row_admx(), label_main(-1), label_admx(), scale_main(0.0f), scale_admx() {}
        MixupData(int32_t _row_main,int32_t _label_main, float _scale_main):
            row_main(_row_main), row_admx(),
            label_main(_label_main), label_admx(),
            scale_main(_scale_main), scale_admx()
        {}
        MixupData(
            int32_t _row_main, int32_t _rows_admx,
            int32_t _label_main, int32_t _labels_admx,
            float _scale_main, float _scale_admx
        ):
            row_main(_row_main), row_admx(_rows_admx),
            label_main(_label_main), label_admx(_labels_admx),
            scale_main(_scale_main), scale_admx(_scale_admx)
        {}
    };

protected:
    const std::string mix_mode;
    NnetChainExampleWriter& example_writer;
    std::string scale_fst_algo;
    bool swap_scales;
    float scale_boost;
    float scale_eps;
    bool max_super;
    int32_t min_shift;
    int32_t max_shift;
    float fixed;
    size_t buff_size;
    int32_t frame_shift;
    std::vector<std::string> exclude_names;
    bool compress;
    bool test_mode;
    rand_gen_t rand_gen;
    uint_distrib_t int_distrib;
    int_distrib_t shift_distrib;
    real_distrib_t float_distrib;
    RandomScale scale_distrib;
    eg_to_egs_t eg_to_egs;
    egs_buffer_t egs_buffer;
    FloatCounter scale_count;
    IntCounter shift_count;
    size_t num_mixed;
    size_t num_untouched;
    size_t num_accepted;
    size_t num_wrote;

public:
    ExampleMixer(
        std::string _mix_mode, const std::string& _distrib,
        NnetChainExampleWriter& _example_writer,
        std::string _scale_fst_algo, bool _swap_scales,
        bool _max_super, int32_t _min_shift, int32_t _max_shift,
        float _fixed, size_t _buff_size, int32_t _frame_shift,
        bool _compress, bool _test_mode
    );

protected:
    const std::vector<Index>& FindIndexes(const std::string& _name, const std::vector<NnetIo>& _nnet_io) const;
    GeneralMatrix& FindFeatures(const std::string& _name, std::vector<NnetIo>& _nnet_io) const;
    const GeneralMatrix& FindFeatures(const std::string& _name, const std::vector<NnetIo>& _nnet_io) const;
    GeneralMatrix* FindIVector(std::vector<NnetIo>& _nnet_io) const;
    const GeneralMatrix* FindIVector(const std::vector<NnetIo>& _nnet_io) const;
    void CheckConsistence(const NnetChainExample& _example) const;
    void CheckConsistence(const NnetChainExample& _example1, const NnetChainExample& _example2) const;
    void ScaleGraph(fst_t& _target, float _scale) const;
    void UnionGraphs(const fst_t& _admixture, fst_t& _example) const;
    void FuseGraphs(const fst_t& _admixture, float _admx_scale, fst_t& _example) const;
    void AdmixGlobal(const NnetChainExample& _admixture, float _admx_scale, ExamplePair& _example);
    void FlushGlobal(egs_buffer_t& _buffer);
    void ShiftAndMixup(ExamplePair& _example);

public:
    void AcceptExample(ExamplePair& _example);
    void Finish();
    const FloatCounter& ScaleCount() const {return scale_count;}
    const IntCounter& ShiftCount() const {return shift_count;}
    size_t NumMixed() const {return num_mixed;}
    size_t NumUntouched() const {return num_untouched;}
    size_t NumAccepted() const {return num_accepted;}
    size_t NumWrote() const {return num_wrote;}
};

ExampleMixer::ExampleMixer(
    std::string _mix_mode, const std::string& _distrib,
    NnetChainExampleWriter& _example_writer,
    std::string _scale_fst_algo, bool _swap_scales,
    bool _max_super, int32_t _min_shift, int32_t _max_shift,
    float _fixed, size_t _buff_size, int32_t _frame_shift,
    bool _compress, bool _test_mode
):
    mix_mode(std::move(_mix_mode)), example_writer(_example_writer),
    scale_fst_algo(std::move(_scale_fst_algo)), swap_scales(_swap_scales),
    scale_boost(1.0f), scale_eps(1e-3f),
    max_super(_max_super), min_shift(_min_shift), max_shift(_max_shift),
    fixed(_fixed), buff_size(_buff_size), frame_shift(_frame_shift),
    exclude_names(), compress(_compress && !_test_mode), test_mode(_test_mode),
    rand_gen(), int_distrib(0, 100000), shift_distrib(_min_shift, _max_shift),
    float_distrib(0.0f, 1.0f), scale_distrib(rand_gen, _distrib), eg_to_egs(), egs_buffer(),
    scale_count(), shift_count(), num_mixed(0), num_untouched(0), num_accepted(0), num_wrote(0)
{
    rand_gen.seed(static_cast<unsigned int>(std::time(0)));
    const size_t indx = scale_fst_algo.find(':');
    if (indx != std::string::npos) {
        std::string value = scale_fst_algo.substr(indx + 1);
        if (value.empty()) {
            KALDI_ERR << "Invalid FST scaling parameners string format: \"" << scale_fst_algo << "\".";
        }
        std::vector<std::string> parts;
        boost::split(parts, value, boost::is_any_of(","), boost::token_compress_on);
        scale_fst_algo = scale_fst_algo.substr(0, indx);
        if (scale_fst_algo.empty() || (scale_fst_algo == "noscale")) {
            scale_eps = boost::lexical_cast<float>(parts.at(0));
        } else {
            scale_boost = boost::lexical_cast<float>(parts.at(0));
            if (parts.size() > 1) {
                scale_eps = boost::lexical_cast<float>(parts.at(1));
            }
        }
    }
    exclude_names.emplace_back(std::string("ivector"));
    KALDI_LOG << "mix_mode: " << mix_mode;
    KALDI_LOG << "distrib: " << _distrib;
    KALDI_LOG << "scale_fst_algo: \"" << scale_fst_algo << "\"";
    KALDI_LOG << "swap_scales: " << (swap_scales? "yes": "no");
    KALDI_LOG << "scale_boost: " << scale_boost;
    KALDI_LOG << "scale_eps: " << scale_eps;
    KALDI_LOG << "max_super: " << (max_super? "yes": "no");
    if (mix_mode == "shift") {
        KALDI_LOG << "min_shift: " << min_shift;
        KALDI_LOG << "max_shift: " << max_shift;
    }
    KALDI_LOG << "fixed: " << fixed;
    KALDI_LOG << "buff_size: " << buff_size;
    KALDI_LOG << "frame_shift: " << frame_shift;
    KALDI_LOG << "compress: " << (compress? "yes": "no");
    KALDI_LOG << "test_mode: " << (test_mode? "yes": "no");
}

const std::vector<Index>& ExampleMixer::FindIndexes(const std::string& _name, const std::vector<NnetIo>& _nnet_io) const {
    for (size_t i = 0; i < _nnet_io.size(); ++i) {
        const NnetIo& nnet_io = _nnet_io[i];
        if (nnet_io.name == _name) {
            if ((nnet_io.indexes.size() != nnet_io.features.NumRows())) {
                KALDI_ERR << "Data indexes have wrong dimension " << nnet_io.indexes.size() << " (must be " << nnet_io.features.NumRows() << ").";
            }
            return nnet_io.indexes;
        }
    }
    KALDI_ERR << "Failed to find example indexes with name \"" << _name << "\".";
}

GeneralMatrix& ExampleMixer::FindFeatures(const std::string& _name, std::vector<NnetIo>& _nnet_io) const {
    for (size_t i = 0; i < _nnet_io.size(); ++i) {
        NnetIo& nnet_io = _nnet_io[i];
        if (nnet_io.name == _name) {
            if ((nnet_io.indexes.size() != nnet_io.features.NumRows())) {
                KALDI_ERR << "Data indexes have wrong dimension " << nnet_io.indexes.size() << " (must be " << nnet_io.features.NumRows() << ").";
            }
            return nnet_io.features;
        }
    }
    KALDI_ERR << "Failed to find example features with name \"" << _name << "\".";
}

const GeneralMatrix& ExampleMixer::FindFeatures(const std::string& _name, const std::vector<NnetIo>& _nnet_io) const {
    for (size_t i = 0; i < _nnet_io.size(); ++i) {
        const NnetIo& nnet_io = _nnet_io[i];
        if (nnet_io.name == _name) {
            if ((nnet_io.indexes.size() != nnet_io.features.NumRows())) {
                KALDI_ERR << "Data indexes have wrong dimension " << nnet_io.indexes.size() << " (must be " << nnet_io.features.NumRows() << ").";
            }
            return nnet_io.features;
        }
    }
    KALDI_ERR << "Failed to find example features with name \"" << _name << "\".";
}

GeneralMatrix* ExampleMixer::FindIVector(std::vector<NnetIo>& _nnet_io) const {
    for (size_t i = 0; i < _nnet_io.size(); ++i) {
        NnetIo& nnet_io = _nnet_io[i];
        if (nnet_io.name == "ivector") {
            if ((nnet_io.indexes.size() != nnet_io.features.NumRows())) {
                KALDI_ERR << "I-vector indexes have wrong dimension " << nnet_io.indexes.size() << " (must be " << nnet_io.features.NumRows() << ").";
            }
            return &nnet_io.features;
        }
    }
    return NULL;
}

const GeneralMatrix* ExampleMixer::FindIVector(const std::vector<NnetIo>& _nnet_io) const {
    for (size_t i = 0; i < _nnet_io.size(); ++i) {
        const NnetIo& nnet_io = _nnet_io[i];
        if (nnet_io.name == "ivector") {
            if ((nnet_io.indexes.size() != nnet_io.features.NumRows())) {
                KALDI_ERR << "I-vector indexes have wrong dimension " << nnet_io.indexes.size() << " (must be " << nnet_io.features.NumRows() << ").";
            }
            return &nnet_io.features;
        }
    }
    return NULL;
}

void ExampleMixer::CheckConsistence(const NnetChainExample& _example) const {
    if (_example.outputs.size() > 1) {
        KALDI_ERR << "Examples with multiple outputs are not supported.";
    }
    if (_example.outputs.front().name != "output") {
        KALDI_ERR << "Unexpected example output name \"" << _example.outputs.front().name << " (must be \"output\").";
    }
    const chain::Supervision& superv = _example.outputs.front().supervision;
    if (superv.num_sequences != 1) {
        KALDI_ERR << "Supervision has unexpected number of sequences " << superv.num_sequences << " (must be 1).";
    }
    if (_example.outputs.front().indexes.size() != superv.frames_per_sequence) {
        KALDI_ERR << "Unexpected number of supervision indexes " << _example.outputs.front().indexes.size() << " (must be " << superv.frames_per_sequence << ").";
    }
    if (_example.outputs.front().deriv_weights.Dim() != superv.frames_per_sequence) {
        KALDI_ERR << "Example output deriv_weights has unexpected dimension " << _example.outputs.front().deriv_weights.Dim() << " (must be " << superv.frames_per_sequence << ").";
    }
}

void ExampleMixer::CheckConsistence(const NnetChainExample& _example1, const NnetChainExample& _example2) const {
    CheckConsistence(_example1);
    CheckConsistence(_example2);
    const NnetChainSupervision& exam_sup1 = _example1.outputs.front();
    const NnetChainSupervision& exam_sup2 = _example2.outputs.front();
    if (exam_sup1.indexes != exam_sup2.indexes) {
        KALDI_ERR << "Examples have different indexes.";
    }
    const int32_t fp_seq1 = exam_sup1.supervision.frames_per_sequence;
    const int32_t fp_seq2 = exam_sup2.supervision.frames_per_sequence;
    if (fp_seq1 != fp_seq2) {
        KALDI_ERR << "Examples have different sequences lengths: " << fp_seq1 << " and " << fp_seq2 << ".";
    }
    if ((exam_sup1.supervision.fst.NumStates() == 0) && (exam_sup1.supervision.e2e_fsts.size() != exam_sup2.supervision.e2e_fsts.size())) {
        KALDI_ERR << "Examples have different number of e2e_fsts: " << exam_sup1.supervision.e2e_fsts.size() << " and " << exam_sup2.supervision.e2e_fsts.size() << ".";
    }
}

void ExampleMixer::ScaleGraph(fst_t& _target, float _scale) const {
    typedef fst::MutableArcIterator<fst_t> iter_t;
    const float scale = -(std::log(_scale) * scale_boost);
    for (state_t state = 0; state < _target.NumStates(); ++state) {
        for (iter_t iarc(&_target, state); !iarc.Done(); iarc.Next()) {
            const arc_t& arc = iarc.Value();
            if (_target.Final(arc.nextstate) != wght_t::Zero()) {
                iarc.SetValue(arc_t(arc.ilabel, arc.olabel, wght_t(arc.weight.Value() + scale), arc.nextstate));
            }
        }
    }
}

void ExampleMixer::UnionGraphs(const fst_t& _admixture, fst_t& _example) const {
    fst_t result;
    result.ReserveStates(_example.NumStates() + _admixture.NumStates());
    const state_t start = result.AddState();
    result.SetStart(start);
    std::vector<state_t> stmap(_example.NumStates(), -1);
    for (state_t state = 0; state < _example.NumStates(); ++state) {
        stmap[state] = (state == _example.Start())? start: result.AddState();
        const wght_t& wght = _example.Final(state);
        if (wght != wght_t::Zero()) {
            if (state == _example.Start()) {
                KALDI_WARN << "Final and start states are equal!";
            }
            result.SetFinal(stmap[state], wght);
        }
    }
    for (state_t state = 0; state < _example.NumStates(); ++state) {
        const state_t begin = (state == _example.Start())? start: stmap[state];
        for (iter_t iarc(_example, state); !iarc.Done(); iarc.Next()) {
            const arc_t& arc = iarc.Value();
            const state_t end = (arc.nextstate == _example.Start())? start: stmap[arc.nextstate];
            result.AddArc(begin, arc_t(arc.ilabel, arc.olabel, arc.weight, end));
        }
    }
    stmap.assign(_admixture.NumStates(), -1);
    for (state_t state = 0; state < _admixture.NumStates(); ++state) {
        stmap[state] = (state == _admixture.Start())? start: result.AddState();
        const wght_t& wght = _admixture.Final(state);
        if (wght != wght_t::Zero()) {
            if (state == _admixture.Start()) {
                KALDI_WARN << "Final and start states are equal!";
            }
            result.SetFinal(stmap[state], wght);
        }
    }
    for (state_t state = 0; state < _admixture.NumStates(); ++state) {
        const state_t begin = (state == _admixture.Start())? start: stmap[state];
        for (iter_t iarc(_admixture, state); !iarc.Done(); iarc.Next()) {
            const arc_t& arc = iarc.Value();
            const state_t end = (arc.nextstate == _admixture.Start())? start: stmap[arc.nextstate];
            result.AddArc(begin, arc_t(arc.ilabel, arc.olabel, arc.weight, end));
        }
    }
    _example = result;
}

void ExampleMixer::FuseGraphs(const fst_t& _admixture, float _admx_scale, fst_t& _example) const {
    if (_admx_scale < scale_eps) {
        return;
    } else if ((1.0f - _admx_scale) < scale_eps) {
        _example = _admixture;
        return;
    }
    if (scale_fst_algo.empty() || (scale_fst_algo == "noscale")) {
        UnionGraphs(_admixture, _example);
    } else if (scale_fst_algo == "default") {
        fst_t admixture(_admixture);
        if (swap_scales) {
            ScaleGraph(admixture, 1.0f - _admx_scale);
            ScaleGraph(_example, _admx_scale);
        } else {
            ScaleGraph(admixture, _admx_scale);
            ScaleGraph(_example, 1.0f - _admx_scale);
        }
        UnionGraphs(admixture, _example);
    } else if (scale_fst_algo == "balanced") {
        const double scale_norm = std::sqrt(_admx_scale * (1.0 - _admx_scale));
        const auto main_scale = (float)((1.0 - _admx_scale) / scale_norm);
        const auto admx_scale = (float)(_admx_scale / scale_norm);
        fst_t admixture(_admixture);
        if (swap_scales) {
            ScaleGraph(admixture, main_scale);
            ScaleGraph(_example, admx_scale);
        } else {
            ScaleGraph(admixture, admx_scale);
            ScaleGraph(_example, main_scale);
        }
        UnionGraphs(admixture, _example);
    } else {
        KALDI_ERR << "Unknown FST scaling algorithm ID: \"" << scale_fst_algo << "\".";
    }
}

void ExampleMixer::AdmixGlobal(const NnetChainExample& _admixture, float _admx_scale, ExamplePair& _example) {
    CheckConsistence(*_example.second, _admixture);
    const float exam_scale = 1.0f - _admx_scale;
    GeneralMatrix* ivector = FindIVector(_example.second->inputs);
    if (ivector != NULL) {
        KaldiMatrix ivec_main;
        ivector->GetMatrix(&ivec_main);
        ivec_main.Scale(exam_scale);
        KaldiMatrix ivec_admx;
        FindIVector(_admixture.inputs)->GetMatrix(&ivec_admx);
        ivec_main.AddMat(_admx_scale, ivec_admx);
        *ivector = ivec_main;
        if (compress) {
            ivector->Compress();
        }
    }
    GeneralMatrix& features = FindFeatures("input", _example.second->inputs);
    KaldiMatrix feat_main;
    features.GetMatrix(&feat_main);
    feat_main.Scale(exam_scale);
    KaldiMatrix feat_admx;
    FindFeatures("input", _admixture.inputs).GetMatrix(&feat_admx);
    feat_main.AddMat(_admx_scale, feat_admx);
    features = feat_main;
    if (compress) {
        features.Compress();
    }
    NnetChainSupervision& exam_nn_sup = _example.second->outputs.front();
    const NnetChainSupervision& admx_nn_sup = _admixture.outputs.front();
    if (max_super) {
        if (_admx_scale > 0.5f) {
            exam_nn_sup = admx_nn_sup;
        }
    } else {
        exam_nn_sup.deriv_weights.Scale(exam_scale);
        exam_nn_sup.deriv_weights.AddVec(_admx_scale, admx_nn_sup.deriv_weights);
        chain::Supervision &exam_sup = exam_nn_sup.supervision;
        const chain::Supervision &admx_sup = admx_nn_sup.supervision;
        exam_sup.weight = exam_sup.weight * exam_scale + admx_sup.weight * _admx_scale;
        if (admx_sup.fst.NumStates() == 0) {
            for (size_t i = 0; i < exam_sup.e2e_fsts.size(); ++i) {
                FuseGraphs(admx_sup.e2e_fsts.at(i), _admx_scale, exam_sup.e2e_fsts.at(i));
            }
        } else {
            FuseGraphs(admx_sup.fst, _admx_scale, exam_sup.fst);
        }
    }
    scale_count += _admx_scale;
}

void ExampleMixer::FlushGlobal(egs_buffer_t& _buffer) {
    egs_buffer_t buffer(_buffer.size());
    for (size_t i = 0; i < _buffer.size(); ++i) {
        const ExamplePair& pair = _buffer[i];
        buffer[i] = std::make_pair(pair.first, ExamplePtr(new NnetChainExample(*pair.second)));
    }
    for (size_t i = 0; i < _buffer.size(); ++i) {
        ExamplePair& example = _buffer.at(i);
        const bool mixup = ((float_distrib(rand_gen) > fixed) && (_buffer.size() > 1));
        if (mixup) {
            size_t indx = int_distrib(rand_gen) % buffer.size();
            if (indx == i) {
                indx = (indx == (buffer.size() - 1)) ? indx - 1 : indx + 1;
            }
            const float scale = scale_distrib();
            AdmixGlobal(*buffer.at(indx).second, scale, example);
            ++num_mixed;
        } else {
            ++num_untouched;
        }
        if (frame_shift != 0) {
            ShiftChainExampleTimes(frame_shift, exclude_names, example.second.get());
        }
        example_writer.Write(example.first, *example.second);
        ++num_wrote;
    }
}

void ExampleMixer::ShiftAndMixup(ExamplePair& _example) {
    CheckConsistence(*_example.second);
    GeneralMatrix& features = FindFeatures("input", _example.second->inputs);
    KaldiMatrix feat_main;
    features.GetMatrix(&feat_main);
    KaldiMatrix feat_admx(feat_main.NumRows(), feat_main.NumCols());
    const int32_t shift = ((float_distrib(rand_gen) > 0.5)? 1: -1) * shift_distrib(rand_gen);
    for (int32_t i = 0; i < feat_main.NumRows(); ++i) {
        int32_t admx_indx = i + shift;
        if (admx_indx < 0) {
            admx_indx = 0;
        } else if (admx_indx >= feat_main.NumRows()) {
            admx_indx = feat_main.NumRows() - 1;
        }
        feat_admx.Row(i).CopyRowFromMat(feat_main, admx_indx);
    }
    const float admx_scale = scale_distrib();
    const float exam_scale = 1.0f - admx_scale;
    feat_main.Scale(exam_scale);
    feat_main.AddMat(admx_scale, feat_admx);
    features = feat_main;
    if (compress) {
        features.Compress();
    }
    scale_count += admx_scale;
    shift_count += shift;
}

void ExampleMixer::AcceptExample(ExamplePair& _example) {
    if (mix_mode == "global") {
        egs_buffer_t &buffer = eg_to_egs[_example.second.get()];
        if (buffer.empty()) {
            buffer.reserve(buff_size);
        }
        buffer.push_back(_example);
        if (buffer.size() == buff_size) {
            egs_buffer_t buff_copy = buffer;
            eg_to_egs.erase(_example.second.get());
            FlushGlobal(buff_copy);
        }
    } else if (mix_mode == "shift") {
        const bool mixup = (float_distrib(rand_gen) > fixed);
        if (mixup) {
            ShiftAndMixup(_example);
            ++num_mixed;
        } else {
            ++num_untouched;
        }
        example_writer.Write(_example.first, *_example.second);
        ++num_wrote;
    } else {
        KALDI_ERR << "Unknown mixup mode: \"" << mix_mode << "\"";
    }
    ++num_accepted;
}

void ExampleMixer::Finish() {
    if (mix_mode == "global") {
        while (!eg_to_egs.empty()) {
            egs_buffer_t buffer = eg_to_egs.begin()->second;
            eg_to_egs.erase(eg_to_egs.begin());
            if (!buffer.empty()) {
                FlushGlobal(buffer);
            }
        }
    }
}

} }

bool AsBool(const char* _value) {
    if (_value == NULL) {
        KALDI_ERR << "Pointer to bool string is nullptr.";
    }
    std::string value(_value);
    boost::to_lower(value);
    if ((value == "true") || (value == "yes") || (value == "on") || (value == "1")) {
        return true;
    } else {
        return false;
    }
}

// --test-mode=true --distrib=uniform:0.1,0.4 --frame-shift=1 ark:/media/work/coding/data/mgb3/train_data/cegs.10.ark ark:/dev/null
// --test-mode=true --distrib=uniform:0.1,0.4 --frame-shift=1 ark:/mnt/TOSHIBA/khokhlov/coding/temp/i-vect/temp/mixup/cegs.10.ark ark:/dev/null
// --test-mode=true --distrib=uniform:0.0,0.4 --scale-fst-algo=noscale:0.2 --frame-shift=1 ark:/mnt/TOSHIBA/khokhlov/coding/temp/i-vect/temp/mixup/cegs.10.ark ark:/dev/null
// --test-mode=true --distrib=uniform:0.0,0.4 --scale-fst-algo=default --swap-scales=true --frame-shift=1 ark:/mnt/TOSHIBA/khokhlov/coding/temp/i-vect/temp/mixup/cegs.10.ark ark:/dev/null
// --test-mode=true --distrib=uniform:0.0,0.4 --scale-fst-algo=default:1.5 --frame-shift=1 ark:/mnt/TOSHIBA/khokhlov/coding/temp/i-vect/temp/mixup/cegs.10.ark ark:/dev/null
// --test-mode=true --distrib=uniform:0.0,0.1 --scale-fst-algo=default:1.5,0.1 --frame-shift=1 ark:/mnt/TOSHIBA/khokhlov/coding/temp/i-vect/temp/mixup/cegs.10.ark ark:/dev/null
// --test-mode=true --distrib=uniform:0.1,0.7 --scale-fst-algo=balanced --frame-shift=1 ark:/mnt/TOSHIBA/khokhlov/coding/temp/i-vect/temp/mixup/cegs.10.ark ark:/dev/null
// --test-mode=true --distrib=uniform:0.0,1.0 --scale-fst-algo=balanced:1.5 --frame-shift=1 ark:/mnt/TOSHIBA/khokhlov/coding/temp/i-vect/temp/mixup/cegs.10.ark ark:/dev/null
// --test-mode=true --distrib=uniform:0.9,1.0 --scale-fst-algo=balanced:1.5,0.2 --frame-shift=1 ark:/mnt/TOSHIBA/khokhlov/coding/temp/i-vect/temp/mixup/cegs.10.ark ark:/dev/null
// --test-mode=true --distrib=beta:0.5 --max-super=false --frame-shift=1 ark:/mnt/TOSHIBA/khokhlov/coding/temp/i-vect/temp/mixup/cegs.10.ark ark:/dev/null
// --test-mode=true --distrib=beta:0.5 --max-super=true --frame-shift=1 ark:/mnt/TOSHIBA/khokhlov/coding/temp/i-vect/temp/mixup/cegs.10.ark ark:/dev/null

// --mix-mode=shift --test-mode=true --distrib=uniform:0.1,0.4 --min-shift=2 --max-shift=5 --frame-shift=1 ark:/mnt/TOSHIBA/khokhlov/coding/temp/i-vect/temp/mixup/cegs.10.ark ark,t:/mnt/TOSHIBA/khokhlov/coding/temp/i-vect/temp/mixup/cegs.10.mix.ark
// --mix-mode=global --test-mode=true --distrib=beta:0.01 --frame-shift=1 ark:/media/work/coding/data/mgb3/train_data/cegs.10.ark ark:/media/work/coding/data/mgb3/train_data/cegs.10.mix.ark
// --mix-mode=global --distrib=beta2:0.5 --buff_size=8000 ark:/media/work/coding/data/mgb3/train_data/cegs.1.ark ark:/dev/null
// --mix-mode=global --distrib=beta2:0.5 --buff_size=8000 ark:/media/work/coding/data/mgb3/train_data/cegs.1.ark ark,t:/media/work/coding/data/mgb3/train_data/cegs.1.mix.ark

// --frame-shift=1 ark:/mnt/TOSHIBA/khokhlov/coding/mixup_github/temp/cegs.1.ark ark:/dev/null

int main(int argc, char *argv[]) {
    try {
        using namespace kaldi;
        using namespace kaldi::nnet3;

        const char *usage =
            "Usage:  nnet3-chain-mixup-egs [options] <egs-rspecifier> <egs-wspecifier>\n"
            "\n"
            "e.g.\n"
            "nnet3-chain-mixup-egs ark:train.egs ark:mixup.egs\n";
        ParseOptions po(usage);

        std::string mix_mode("global");
        const char* env_var = getenv("MIXUP_MIX_MODE");
        if (env_var != NULL) {
            mix_mode = env_var;
        }
        po.Register("mix-mode", &mix_mode, "Mixup mode (\"global\", \"shift\") MIXUP_MIX_MODE");

        std::string distrib("uniform:0.0,0.5");
        env_var = getenv("MIXUP_DISTRIB");
        if (env_var != NULL) {
            distrib = env_var;
        }
        po.Register("distrib", &distrib, "Mixup scaling factors distribution (\"uniform:min,max\", \"beta:alpha\", \"beta2:alpha\") MIXUP_DISTRIB");

        std::string scale_fst_algo;
        env_var = getenv("MIXUP_SCALE_FST_ALGO");
        if (env_var != NULL) {
            scale_fst_algo = env_var;
        }
        po.Register("scale-fst-algo", &scale_fst_algo, "Scale supervision FSTs algorithm (\"default[:scale[,eps]]\", \"balanced[:scale[,eps]]\") MIXUP_SCALE_FST_ALGO");

        bool swap_scales = false;
        env_var = getenv("MIXUP_SWAP_SCALES");
        if (env_var != NULL) {
            swap_scales = AsBool(env_var);
        }
        po.Register("swap-scales", &swap_scales, "Swap supervision FST scales MIXUP_SWAP_SCALES");

        bool max_super = false;
        env_var = getenv("MIXUP_MAX_SUPER");
        if (env_var != NULL) {
            max_super = AsBool(env_var);
        }
        po.Register("max-super", &max_super, "Get supervision from example with maximum scale MIXUP_MAX_SUPER");

        int32_t min_shift = 1;
        env_var = getenv("MIXUP_MIN_SHIFT");
        if (env_var != NULL) {
            min_shift = boost::lexical_cast<int32_t>(env_var);
        }
        po.Register("min-shift", &min_shift, "Minimum sequence shift size (shift mode) MIXUP_MIN_SHIFT");

        int32_t max_shift = 3;
        env_var = getenv("MIXUP_MAX_SHIFT");
        if (env_var != NULL) {
            max_shift = boost::lexical_cast<int32_t>(env_var);
        }
        po.Register("max-shift", &max_shift, "Maximum sequence shift size (shift mode) MIXUP_MAX_SHIFT");

        float fixed = 0.10;
        env_var = getenv("MIXUP_FIXED");
        if (env_var != NULL) {
            fixed = boost::lexical_cast<float>(env_var);
        }
        po.Register("fixed", &fixed, "The portion of the data to leave untouched MIXUP_FIXED");

        int32_t buff_size = 500;
        env_var = getenv("MIXUP_BUFF_SIZE");
        if (env_var != NULL) {
            buff_size = boost::lexical_cast<int32_t>(env_var);
        }
        po.Register("buff-size", &buff_size, "Buffer size for data shuffling (global mode) MIXUP_BUFF_SIZE");

        int32_t frame_shift = 0;
        env_var = getenv("MIXUP_FRAME_SHIFT");
        if (env_var != NULL) {
            frame_shift = boost::lexical_cast<int32_t>(env_var);
        }
        po.Register("frame-shift", &frame_shift, "Allows you to shift time values in the supervision data (excluding iVector data) - useful in augmenting data.  Note, the outputs will remain at the closest exact multiples of the frame subsampling factor MIXUP_FRAME_SHIFT");

        int32_t compress = 0;
        env_var = getenv("MIXUP_COMPRESS");
        if (env_var != NULL) {
            compress = boost::lexical_cast<int32_t>(env_var);
        }
        po.Register("compress", &compress, "Compress features and i-vectors MIXUP_COMPRESS");

        bool test_mode = false;
        po.Register("test-mode", &test_mode, "Self testing mode");

        po.Read(argc, argv);
        if (po.NumArgs() < 2) {
            po.PrintUsage();
            exit(1);
        }
        if (mix_mode == "shift") {
            if (min_shift < 1) {
                KALDI_ERR << "min_num must be greater or equal 1";
            }
        }
        std::string examples_rspecifier = po.GetArg(1);
        std::string examples_wspecifier = po.GetArg(2);

        SequentialNnetChainExampleReader example_reader(examples_rspecifier);
        NnetChainExampleWriter example_writer(examples_wspecifier);

        ExampleMixer mixer(
            mix_mode, distrib, example_writer, scale_fst_algo, swap_scales, max_super,
            min_shift, max_shift, fixed, (size_t) buff_size, frame_shift, compress != 0, test_mode != 0
        );
        size_t num_read = 0;
        for (; !example_reader.Done(); example_reader.Next(), num_read++) {
            const NnetChainExample& example = example_reader.Value();
            ExamplePair ex_pair(example_reader.Key(), ExamplePtr(new NnetChainExample(example)));
            mixer.AcceptExample(ex_pair);
        }
        mixer.Finish();

        const FloatCounter& scale_count = mixer.ScaleCount();
        const IntCounter& shift_count = mixer.ShiftCount();
        const size_t num_mixed = mixer.NumMixed();
        const size_t num_untouched = mixer.NumUntouched();
        const size_t num_accepted = mixer.NumAccepted();
        const size_t num_wrote = mixer.NumWrote();

        KALDI_LOG << "Num read: " << num_read << " examples";
        KALDI_LOG << "Num accepted: " << num_accepted << " examples ( " << (100.0 * num_accepted / num_read) << " % of read )";
        KALDI_LOG << "Num wrote: " << num_wrote << " examples ( " << (100.0 * num_wrote / num_read) << " % of read )";
        KALDI_LOG << "Num mixed: " << num_mixed << " examples ( " << (100.0 * num_mixed / num_accepted) << " % of accepted )";
        KALDI_LOG << "Num untouched: " << num_untouched << " examples ( " << (100.0 * num_untouched / num_accepted) << " % of accepted )";
        KALDI_LOG << "Average scale: " << (boost::format("%.4f ( min: %.4f, max: %.4f )") % scale_count.Average() % scale_count.Minimum() % scale_count.Maximum()).str();
        if (mix_mode == "shift") {
            KALDI_LOG << "Average shift: " << (boost::format("%.4f ( min: %.4f, max: %.4f )") % shift_count.Average() % shift_count.Minimum() % shift_count.Maximum()).str();
        }

        return (num_wrote == 0 ? 1 : 0);
    } catch(const std::exception &e) {
        std::cerr << e.what() << '\n';
        return -1;
    }
}
