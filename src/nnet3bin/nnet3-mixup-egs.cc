// Copyright 2020 Speech Technology Center www.speechpro.com
//
// Licensed under the Apache License, Version 2.0 http://www.apache.org/licenses/LICENSE-2.0
//
// This code is provided *as is* basis, without warranties or conditions of any kind.
////////////////////////////////////////////////////////////////////////////////////////////
// If you find this code useful for your research or production, please cite our paper:
//
// @inproceedings{Medennikov_mixup2018,
//   author={Ivan Medennikov and Yuri Khokhlov and Aleksei Romanenko and Dmitry Popov and Natalia Tomashenko and Ivan Sorokin and Alexander Zatvornitskiy},
//   title={An Investigation of Mixup Training Strategies for Acoustic Models in ASR},
//   year=2018,
//   booktitle={Proc. Interspeech 2018},
//   pages={2903--2907},
//   doi={10.21437/Interspeech.2018-2191},
//   url={http://dx.doi.org/10.21437/Interspeech.2018-2191}
// }
//
// (corresponding author: medennikov@speechpro.com)
////////////////////////////////////////////////////////////////////////////////////////////


#include <limits>
#include <algorithm>
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
#include <utility>
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "nnet3/nnet-example.h"

namespace kaldi { namespace nnet3 {

typedef Matrix<BaseFloat> KaldiMatrix;
typedef Vector<BaseFloat> KaldiVector;
typedef SparseMatrix<BaseFloat> KaldiSparMatrix;
typedef SparseVector<BaseFloat> KaldiSparVector;
typedef SubMatrix<BaseFloat> KaldiSubMatrix;
typedef SubVector<BaseFloat> KaldiSubVector;
typedef boost::shared_ptr<NnetExample> ExamplePtr;
typedef std::pair<std::string, ExamplePtr> ExamplePair;

class LabelsMap {
protected:
    std::map<int32_t, int32_t> lmap;

public:
    explicit LabelsMap(const std::string& _labels_map) : lmap() {
        if (!_labels_map.empty()) {
            std::vector<std::string> parts;
            boost::split(parts, _labels_map, boost::is_any_of(" \t,;|/"), boost::token_compress_on);
            for (const auto &part : parts) {
                std::vector<std::string> pair;
                boost::split(pair, part, boost::is_any_of(":"), boost::token_compress_on);
                if (pair.size() != 2) {
                    KALDI_ERR << "Wrong labels map format: \"" << _labels_map << "\"";
                }
                lmap.insert(std::make_pair(boost::lexical_cast<int32_t>(pair[0]), boost::lexical_cast<int32_t>(pair[1])));
            }
        }
    }
    bool empty() const { return lmap.empty(); }
    int32_t operator()(int32_t _label) const {
        auto iter = lmap.find(_label);
        return (iter == lmap.end()) ? _label : iter->second;
    }
    std::string to_string() const {
        std::stringstream strstrm;
        for (auto& pair : lmap) {
            if (strstrm.tellp() > 0) {
                strstrm << ",";
            }
            strstrm << pair.first << ":" << pair.second;
        }
        return strstrm.str();
    }
};

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

class IScaleTransform {
public:
    virtual float operator()(float _scale) const = 0;
};

class SigmoidTransform : public IScaleTransform {
protected:
    double k;
    double denom;

public:
    explicit SigmoidTransform(double _k): IScaleTransform(), k(_k), denom(2.0 * std::tanh(_k)) {}
    float operator()(float _scale) const override;
};

float SigmoidTransform::operator()(float _scale) const {
    return (float)(0.5 + std::tanh(k * (2.0 * _scale - 1)) / denom);
}

class ScaleTransform : public IScaleTransform {
protected:
    typedef boost::shared_ptr<IScaleTransform> transform_t;

protected:
    transform_t transform;

public:
    explicit ScaleTransform(const std::string& _transform);
    float operator()(float _scale) const override;
};

ScaleTransform::ScaleTransform(const std::string& _transform): IScaleTransform(), transform() {
    if (!_transform.empty()) {
        std::string trans_name;
        float value1 = 0.0f;
        const size_t indx = _transform.find(':');
        if (indx != std::string::npos) {
            std::string value = _transform.substr(indx + 1);
            if (value.empty()) {
                KALDI_ERR << "Invalid transform functions parameners string format: \"" << _transform << "\".";
            }
            std::vector<std::string> parts;
            boost::split(parts, value, boost::is_any_of(","), boost::token_compress_on);
            value1 = boost::lexical_cast<float>(parts.at(0));
            trans_name = _transform.substr(0, indx);
        }
        if (trans_name == "sigmoid") {
            transform.reset(new SigmoidTransform(value1));
        } else {
            KALDI_ERR << "Unknown type of transform \"" << _transform << "\".";
        }
    }
}

float ScaleTransform::operator()(float _scale) const {
    if (transform == nullptr) {
        return _scale;
    } else {
        return (*transform)(_scale);
    }
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

class ExampleMixer {
protected:
    typedef boost::random::mt19937 rand_gen_t;
    typedef boost::random::uniform_int_distribution<int32_t> int_distrib_t;
    typedef boost::random::uniform_int_distribution<size_t> uint_distrib_t;
    typedef boost::random::uniform_real_distribution<float> real_distrib_t;
    typedef std::vector<ExamplePair> egs_buffer_t;
    typedef unordered_map<NnetExample*, egs_buffer_t, NnetExampleStructureHasher, NnetExampleStructureCompare> eg_to_egs_t;

protected:
    struct MixupData {
        int32_t row_main;
        int32_t row_admx;
        int32_t label_main;
        int32_t label_admx;
        float scale;
        MixupData(): row_main(0), row_admx(0), label_main(-1), label_admx(-1), scale(0.0f) {}
        MixupData(int32_t _row_main, int32_t _row_admx, int32_t _label_main, int32_t _label_admx, float _scale):
            row_main(_row_main), row_admx(_row_admx), label_main(_label_main), label_admx(_label_admx), scale(_scale)
        {}
    };

protected:
    const std::string mix_mode;
    NnetExampleWriter& example_writer;
    size_t min_num;
    size_t max_num;
    int32_t min_shift;
    int32_t max_shift;
    float fixed_egs;
    float fixed_frames;
    size_t left_range;
    size_t right_range;
    size_t buff_size;
    bool mix_ivect;
    bool mix_feats;
    bool mix_labels;
    const LabelsMap& labels_map;
    bool compress;
    bool test_mode;
    rand_gen_t rand_gen;
    uint_distrib_t int_distrib;
    uint_distrib_t num_distrib;
    int_distrib_t shift_distrib;
    real_distrib_t float_distrib;
    RandomScale scale_distrib;
    ScaleTransform transform;
    eg_to_egs_t eg_to_egs;
    egs_buffer_t egs_buffer;
    FloatCounter scale_count;
    IntCounter shift_count;
    IntCounter adnum_count;
    IntCounter left_count;
    IntCounter right_count;
    size_t num_mixed;
    size_t num_untouched;
    size_t num_accepted;
    size_t num_wrote;

public:
    ExampleMixer(
        std::string _mix_mode, const std::string& _distrib, const std::string& _transform,
        NnetExampleWriter& _example_writer, size_t _min_num, size_t _max_num,
        int32_t _min_shift, int32_t _max_shift, float _fixed_egs, float _fixed_frames,
        size_t _left_range, size_t _right_range, size_t _buff_size,
        bool _mix_ivect, bool _mix_feats, bool _mix_labels, const LabelsMap& _labels_map,
        bool _compress, bool _test_mode
    );

protected:
    const std::vector<Index>& FindIndexes(const std::string& _name, const std::vector<NnetIo>& _nnet_io) const;
    GeneralMatrix& FindFeatures(const std::string& _name, std::vector<NnetIo>& _nnet_io) const;
    GeneralMatrix* FindIVector(std::vector<NnetIo>& _nnet_io) const;
    void MixupLocal(ExamplePair& _example);
    void AdmixGlobal(const std::vector<float>& _adm_scales, const std::vector<ExamplePtr>& _admixtures, float _exam_scale, ExamplePair& _example);
    void FlushGlobal(egs_buffer_t& _buffer);
    void FlushClass(egs_buffer_t& _buffer);
    void ShiftAndMixup(ExamplePair& _example);

public:
    void AcceptExample(ExamplePair& _example);
    void Finish();
    const FloatCounter& ScaleCount() const {return scale_count;}
    const IntCounter& ShiftCount() const {return shift_count;}
    const IntCounter& AdmixNumCount() const {return adnum_count;}
    const IntCounter& LeftCount() const {return left_count;}
    const IntCounter& RightCount() const {return right_count;}
    size_t NumMixed() const {return num_mixed;}
    size_t NumUntouched() const {return num_untouched;}
    size_t NumAccepted() const {return num_accepted;}
    size_t NumWrote() const {return num_wrote;}
};

ExampleMixer::ExampleMixer(
    std::string _mix_mode, const std::string& _distrib, const std::string& _transform,
    NnetExampleWriter& _example_writer, size_t _min_num, size_t _max_num,
    int32_t _min_shift, int32_t _max_shift, float _fixed_egs, float _fixed_frames,
    size_t _left_range, size_t _right_range, size_t _buff_size,
    bool _mix_ivect, bool _mix_feats, bool _mix_labels, const LabelsMap& _labels_map,
    bool _compress, bool _test_mode
):
    mix_mode(std::move(_mix_mode)), transform(_transform),
    example_writer(_example_writer), min_num(_min_num), max_num(_max_num),
    min_shift(_min_shift), max_shift(_max_shift),
    fixed_egs(_fixed_egs), fixed_frames(_fixed_frames),
    left_range(_left_range), right_range(_right_range), buff_size(_buff_size),
    mix_ivect(_mix_ivect), mix_feats(_mix_feats), mix_labels(_mix_labels),
    labels_map(_labels_map), compress(_compress && !_test_mode), test_mode(_test_mode),
    rand_gen(), int_distrib(0, 100000), num_distrib(min_num, max_num),
    shift_distrib(_min_shift, _max_shift), float_distrib(0.0f, 1.0f),
    scale_distrib(rand_gen, _distrib), eg_to_egs(), egs_buffer(),
    scale_count(), shift_count(), adnum_count(), left_count(), right_count(),
    num_mixed(0), num_untouched(0), num_accepted(0), num_wrote(0)
{
    rand_gen.seed(static_cast<unsigned int>(std::time(0)));
    KALDI_LOG << "mix_mode: " << mix_mode;
    KALDI_LOG << "distrib: " << _distrib;
    KALDI_LOG << "transform: " << _transform;
    if (mix_mode == "global") {
        KALDI_LOG << "min_num: " << min_num;
        KALDI_LOG << "max_num: " << max_num;
    }
    if (mix_mode == "shift") {
        KALDI_LOG << "min_shift: " << min_shift;
        KALDI_LOG << "max_shift: " << max_shift;
    }
    KALDI_LOG << "fixed_egs: " << fixed_egs;
    KALDI_LOG << "fixed_frames: " << fixed_frames;
    if (mix_mode == "local") {
        KALDI_LOG << "left_range: " << left_range;
        KALDI_LOG << "right_range: " << right_range;
    }
    KALDI_LOG << "buff_size: " << buff_size;
    KALDI_LOG << "mix_ivect: " << (mix_ivect? "yes": "no");
    KALDI_LOG << "mix_feats: " << (mix_feats? "yes": "no");
    KALDI_LOG << "mix_labels: " << (mix_labels? "yes": "no");
    KALDI_LOG << "labels_map: " << labels_map.to_string();
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
    return nullptr;
}

void ExampleMixer::MixupLocal(ExamplePair& _example) {
    KaldiMatrix test_feats_org;
    KaldiMatrix test_labels_org;
    if (test_mode) {
        FindFeatures("input", _example.second->io).GetMatrix(&test_feats_org);
        FindFeatures("output", _example.second->io).GetMatrix(&test_labels_org);
    }
    std::vector<NnetIo>& in_out = _example.second->io;
    const std::vector<Index>& in_indx = FindIndexes("input", in_out);
    GeneralMatrix& features_g = FindFeatures("input", in_out);
    const std::vector<Index>& out_indx = FindIndexes("output", in_out);;
    GeneralMatrix& labels_g = FindFeatures("output", in_out);
    KaldiMatrix features_org;
    features_g.GetMatrix(&features_org);
    const KaldiSparMatrix& labels_org = labels_g.GetSparseMatrix();
    KaldiMatrix features_mix(features_org.NumRows(), features_org.NumCols(), kSetZero);
    KaldiSparMatrix labels_mix(labels_org.NumRows(), labels_org.NumCols());
    std::vector<MixupData> mixup_data((size_t) features_org.NumRows());
    for (size_t row_main = 0; row_main < mixup_data.size(); ++row_main) {
        int32_t shift = 0;
        if (float_distrib(rand_gen) > 0.5f) {
            shift = -(int)(int_distrib(rand_gen) % left_range + 1);
            left_count += shift;
        } else {
            shift = (int)(int_distrib(rand_gen) % right_range + 1);
            right_count += shift;
        }
        shift_count += shift;
        int32_t row_admx = int32_t(row_main) + shift;
        row_admx = std::max(0, row_admx);
        row_admx = std::min(features_org.NumRows() - 1, row_admx);
        const int32_t time = in_indx.at(row_main).t;
        int32_t label_main = -1;
        int32_t label_admx = -1;
        if ((time >= 0) && (time < labels_org.NumRows())) {
            label_main = time;
            label_admx = time + shift;
            label_admx = std::max(0, label_admx);
            label_admx = std::min(labels_org.NumRows() - 1, label_admx);
            shift = label_admx - label_main;
            row_admx = int32_t(row_main) + shift;
        }
        const float scale = scale_distrib();
        mixup_data.at(row_main) = MixupData((int32_t) row_main, row_admx, label_main, label_admx, scale);
        scale_count += scale;
    }
    for (const auto & data : mixup_data) {
        const KaldiSubVector frame_main(features_org, data.row_main);
        const KaldiSubVector frame_admx(features_org, data.row_admx);
        KaldiSubVector frame_dest(features_mix, data.row_main);
        frame_dest.AddVec((1.0f - data.scale), frame_main);
        frame_dest.AddVec(data.scale, frame_admx);
        if (data.label_main >= 0) {
            typedef std::pair<int32_t, float> label_t;
            typedef std::map<int32_t, float> labels_t;
            labels_t labels;
            const KaldiSparVector& labls_main = labels_org.Row(data.label_main);
            for (int32_t j = 0; j < labls_main.NumElements(); ++j) {
                const label_t& label = labls_main.GetElement(j);
                labels.insert(std::make_pair(label.first, (1.0f - data.scale) * label.second));
            }
            const KaldiSparVector& labls_admx = labels_org.Row(data.label_admx);
            for (int32_t j = 0; j < labls_admx.NumElements(); ++j) {
                const label_t& label = labls_admx.GetElement(j);
                auto iter = labels.find(label.first);
                if (iter == labels.end()) {
                    labels.insert(std::make_pair(label.first, data.scale * label.second));
                } else {
                    iter->second += data.scale * label.second;
                }
            }
            KaldiSparVector labls_dest(labels_org.NumCols(), std::vector<label_t>(labels.begin(), labels.end()));
            labels_mix.SetRow(data.label_main, labls_dest);
        }
    }
    features_g = features_mix;
    if (compress) {
        features_g.Compress();
    }
    labels_g = labels_mix;
    if (test_mode) {
        {
            KaldiMatrix feats_mix;
            FindFeatures("input", _example.second->io).GetMatrix(&feats_mix);
            for (const auto & data : mixup_data) {
                KaldiVector row_main(test_feats_org.Row(data.row_main));
                KaldiVector row_admx(test_feats_org.Row(data.row_admx));
                row_main.Scale(1.0f - data.scale);
                row_main.AddVec(data.scale, row_admx);
                KaldiSubVector mixed(feats_mix.Row(data.row_main));
                row_main.AddVec(-1.0f, mixed);
                row_main.ApplyAbs();
                const float value = row_main.Max();
                KALDI_ASSERT(value < 1e-9);
            }
        }
        {
            KaldiMatrix labels_mix;
            FindFeatures("output", _example.second->io).GetMatrix(&labels_mix);
            for (const auto & data : mixup_data) {
                if (data.label_main < 0) {
                    continue;
                }
                KaldiVector row_main(test_labels_org.Row(data.label_main));
                KaldiVector row_admx(test_labels_org.Row(data.label_admx));
                row_main.Scale(1.0f - data.scale);
                row_main.AddVec(data.scale, row_admx);
                KaldiSubVector mixed(labels_mix.Row(data.label_main));
                row_main.AddVec(-1.0f, mixed);
                row_main.ApplyAbs();
                const float value = row_main.Max();
                KALDI_ASSERT(value < 1e-9);
            }
        }
    }
}

void ExampleMixer::AdmixGlobal(const std::vector<float>& _adm_scales, const std::vector<ExamplePtr>& _admixtures, float _exam_scale, ExamplePair& _example) {
    if (test_mode) {
        const float diff = std::fabs(1.0f - std::accumulate(_adm_scales.begin(), _adm_scales.end(), _exam_scale));
        KALDI_ASSERT(diff < 1e-6);
    }
    if (_admixtures.size() != _adm_scales.size()) {
        KALDI_ERR << "Wrong admixtures vector size " << _admixtures.size() << " (must be " << _adm_scales.size() << ").";
    }
    KaldiMatrix test_feats_org;
    KaldiMatrix test_ivect_org;
    KaldiMatrix test_labels_org;
    if (test_mode) {
        FindFeatures("input", _example.second->io).GetMatrix(&test_feats_org);
        if (FindIVector(_example.second->io) != nullptr) {
            FindIVector(_example.second->io)->GetMatrix(&test_ivect_org);
        }
        FindFeatures("output", _example.second->io).GetMatrix(&test_labels_org);
    }
    GeneralMatrix* ivector = FindIVector(_example.second->io);
    if (mix_ivect && (ivector != nullptr)) {
        KaldiMatrix ivec_main;
        ivector->GetMatrix(&ivec_main);
        ivec_main.Scale(_exam_scale);
        for (size_t i = 0; i < _adm_scales.size(); ++i) {
            KaldiMatrix ivec_admx;
            FindIVector(_admixtures[i]->io)->GetMatrix(&ivec_admx);
            ivec_main.AddMat(_adm_scales[i], ivec_admx);
        }
        *ivector = ivec_main;
        if (compress) {
            ivector->Compress();
        }
    }
    if (mix_feats) {
        GeneralMatrix &features = FindFeatures("input", _example.second->io);
        KaldiMatrix feat_main;
        features.GetMatrix(&feat_main);
        feat_main.Scale(_exam_scale);
        for (size_t i = 0; i < _adm_scales.size(); ++i) {
            KaldiMatrix feat_admx;
            FindFeatures("input", _admixtures[i]->io).GetMatrix(&feat_admx);
            feat_main.AddMat(_adm_scales[i], feat_admx);
        }
        features = feat_main;
        if (compress) {
            features.Compress();
        }
    }
    float exam_scale = transform(_exam_scale);
    std::vector<float> adm_scales(_adm_scales);
    for (float & adm_scale : adm_scales) {
        adm_scale = transform(adm_scale);
    }
    const float scale_norm = std::accumulate(adm_scales.begin(), adm_scales.end(), exam_scale);
    exam_scale /= scale_norm;
    std::transform(adm_scales.begin(), adm_scales.end(), adm_scales.begin(), [scale_norm](float _value) -> float { return _value / scale_norm; });
    if (mix_labels) {
        GeneralMatrix &labels = FindFeatures("output", _example.second->io);
        KaldiSparMatrix labels_main(labels.GetSparseMatrix());
        labels_main.Scale(exam_scale);
        for (size_t i = 0; i < adm_scales.size(); ++i) {
            KaldiSparMatrix labels_admx(FindFeatures("output", _admixtures[i]->io).GetSparseMatrix());
            labels_admx.Scale(adm_scales[i]);
            for (int32_t j = 0; j < labels_main.NumRows(); ++j) {
                typedef std::pair<int32_t, float> label_t;
                typedef std::map<int32_t, float> labels_t;
                labels_t lab_set;
                const KaldiSparVector &lab_main = labels_main.Row(j);
                for (int32_t k = 0; k < lab_main.NumElements(); ++k) {
                    lab_set.insert(lab_main.GetElement(k));
                }
                const KaldiSparVector &lab_admx = labels_admx.Row(j);
                for (int32_t k = 0; k < lab_admx.NumElements(); ++k) {
                    label_t label = lab_admx.GetElement(k);
                    if (!labels_map.empty()) {
                        label.first = labels_map(label.first);
                    }
                    auto iter = lab_set.find(label.first);
                    if (iter == lab_set.end()) {
                        lab_set.insert(label);
                    } else {
                        iter->second += label.second;
                    }
                }
                KaldiSparVector labls_dest(labels_main.NumCols(), std::vector<label_t>(lab_set.begin(), lab_set.end()));
                labels_main.SetRow(j, labls_dest);
            }
        }
        labels = labels_main;
    }
    scale_count += 1.0 - _exam_scale;
    adnum_count += _adm_scales.size();
    if (test_mode) {
        {
            KaldiMatrix feats_res;
            FindFeatures("input", _example.second->io).GetMatrix(&feats_res);
            feats_res.AddMat(-_exam_scale, test_feats_org);
            for (size_t i = 0; i < _adm_scales.size(); ++i) {
                KaldiMatrix feats_tmp;
                FindFeatures("input", _admixtures[i]->io).GetMatrix(&feats_tmp);
                feats_res.AddMat(-_adm_scales[i], feats_tmp);
            }
            const float value = std::max(std::fabs(feats_res.Max()), std::fabs(feats_res.Min())) / _adm_scales.size();
            KALDI_ASSERT(value < 1e-5);
        }
        if (ivector != nullptr) {
            KaldiMatrix ivect_res;
            FindIVector(_example.second->io)->GetMatrix(&ivect_res);
            ivect_res.AddMat(-_exam_scale, test_ivect_org);
            for (size_t i = 0; i < _adm_scales.size(); ++i) {
                KaldiMatrix ivect_tmp;
                FindIVector(_admixtures[i]->io)->GetMatrix(&ivect_tmp);
                ivect_res.AddMat(-_adm_scales[i], ivect_tmp);
            }
            const float value = std::max(std::fabs(ivect_res.Max()), std::fabs(ivect_res.Min())) / _adm_scales.size();
            KALDI_ASSERT(value < 1e-6);
        }
        {
            KaldiMatrix labels_res;
            FindFeatures("output", _example.second->io).GetMatrix(&labels_res);
            labels_res.AddMat(-exam_scale, test_labels_org);
            for (size_t i = 0; i < adm_scales.size(); ++i) {
                KaldiMatrix labels_tmp;
                FindFeatures("output", _admixtures[i]->io).GetMatrix(&labels_tmp);
                labels_res.AddMat(-adm_scales[i], labels_tmp);
            }
            const float value = std::max(std::fabs(labels_res.Max()), std::fabs(labels_res.Min())) / adm_scales.size();
            KALDI_ASSERT(value < 1e-7);
        }
    }
}

void ExampleMixer::FlushGlobal(egs_buffer_t& _buffer) {
    egs_buffer_t buffer(_buffer.size());
    for (size_t i = 0; i < _buffer.size(); ++i) {
        const ExamplePair& pair = _buffer[i];
        buffer[i] = std::make_pair(pair.first, ExamplePtr(new NnetExample(*pair.second)));
    }
    for (size_t i = 0; i < _buffer.size(); ++i) {
        ExamplePair& example = _buffer.at(i);
        const bool mixup = ((float_distrib(rand_gen) > fixed_egs) && (_buffer.size() > 1));
        if (mixup) {
            const size_t admix_num = (min_num == max_num)? min_num: num_distrib(rand_gen);
            std::vector<float> scales;
            scales.reserve(admix_num);
            float summ = 0.0f;
            for (size_t j = 0; j < admix_num; ++j) {
                const float scale = float_distrib(rand_gen);
                scales.push_back(scale);
                summ += scale;
            }
            const float scale = scale_distrib();
            for (size_t j = 0; j < scales.size(); ++j) {
                scales[j] *= scale / summ;
            }
            if (test_mode) {
                summ = 0.0f;
                for (size_t j = 0; j < scales.size(); ++j) {
                    summ += scales[j];
                }
                const float diff = std::fabs(summ - scale);
                KALDI_ASSERT(diff < 1e-6);
            }
            std::vector<ExamplePtr> admixts;
            admixts.reserve(scales.size());
            for (size_t j = 0; j < scales.size(); ++j) {
                size_t indx = int_distrib(rand_gen) % buffer.size();
                if (indx == i) {
                    indx = (indx == (buffer.size() - 1)) ? indx - 1 : indx + 1;
                }
                admixts.push_back(buffer.at(indx).second);
            }
            AdmixGlobal(scales, admixts, 1.0f - scale, example);
            ++num_mixed;
        } else {
            ++num_untouched;
        }
        example_writer.Write(example.first, *example.second);
        ++num_wrote;
    }
}

void ExampleMixer::FlushClass(egs_buffer_t& _buffer) {
    typedef std::pair<const KaldiSubVector, const KaldiSparVector*> frame_pair_t;
    typedef std::vector<frame_pair_t> frame_arr_t;
    typedef unordered_map<int32_t, frame_arr_t> frame_map_t;
    frame_map_t frame_map;
    egs_buffer_t buffer(_buffer.size());
    for (size_t i = 0; i < _buffer.size(); ++i) {
        const ExamplePair& pair = _buffer[i];
        buffer[i] = std::make_pair(pair.first, ExamplePtr(new NnetExample(*pair.second)));
        NnetExample& example = *buffer[i].second;
        const std::vector<Index>& in_indx = FindIndexes("input", example.io);
        const std::vector<Index>& out_indx = FindIndexes("output", example.io);
        if (out_indx.front().t != 0) {
            KALDI_ERR << "Unexpected time stamp of first label " << out_indx.front().t << " (zero expected).";
        }
        if (out_indx.back().t != (out_indx.size() - 1)) {
            KALDI_ERR << "Unexpected time stamp of last label " << out_indx.back().t << " (must be " << (out_indx.size() - 1) << ").";
        }
        GeneralMatrix& gen_matrix = FindFeatures("input", example.io);
        {
            KaldiMatrix matrix;
            gen_matrix.GetMatrix(&matrix);
            gen_matrix = matrix;
        }
        const KaldiMatrix& matrix = gen_matrix.GetFullMatrix();
        if (matrix.NumRows() != in_indx.size()) {
            KALDI_ERR << "Wrong number of rows in input features matrix " << matrix.NumRows() << " (must be " << in_indx.size() << ").";
        }
        const KaldiSparMatrix& labels = FindFeatures("output", example.io).GetSparseMatrix();
        if (labels.NumRows() != out_indx.size()) {
            KALDI_ERR << "Wrong number of rows in output labels matrix " << labels.NumRows() << " (must be " << out_indx.size() << ").";
        }
        int32_t last_label = -1;
        auto class_iter = frame_map.end();
        for (size_t j = 0; j < in_indx.size(); ++j) {
            const Index& indx = in_indx[j];
            if ((indx.t < 0) || (indx.t >= labels.NumRows())) {
                continue;
            }
            const KaldiSparVector& spvect = labels.Row(indx.t);
            int32_t label = spvect.GetElement(0).first;
            if (spvect.NumElements() > 1) {
                float max_value = spvect.GetElement(0).second;
                for (int32_t k = 1; k < spvect.NumElements(); ++k) {
                    const std::pair<MatrixIndexT, BaseFloat>& item = spvect.GetElement(k);
                    if (max_value < item.second) {
                        label = item.first;
                        max_value = item.second;
                    }
                }
            }
            if (last_label != label) {
                class_iter = frame_map.find(label);
                if (class_iter == frame_map.end()) {
                    class_iter = frame_map.insert(std::make_pair(label, frame_arr_t())).first;
                }
                last_label = label;
            }
            class_iter->second.emplace_back(std::make_pair(matrix.Row((MatrixIndexT) j), &spvect));
        }
    }
    for (auto & pair : _buffer) {
        if (float_distrib(rand_gen) > fixed_egs) {
            NnetExample& example = *pair.second;
            const std::vector<Index>& in_indx = FindIndexes("input", example.io);
            GeneralMatrix& gen_matrix = FindFeatures("input", example.io);
            KaldiMatrix matrix;
            gen_matrix.GetMatrix(&matrix);
            const std::vector<Index>& out_indx = FindIndexes("output", example.io);
            GeneralMatrix& gen_labels = FindFeatures("output", example.io);
            KaldiSparMatrix labels = gen_labels.GetSparseMatrix();
            int32_t last_label = -1;
            auto class_iter = frame_map.end();
            for (auto indx : in_indx) {
                if ((indx.t < 0) || (indx.t >= labels.NumRows())) {
                    continue;
                }
                if (float_distrib(rand_gen) < fixed_frames) {
                    continue;
                }
                KaldiSubVector frame_row1 = matrix.Row(indx.t);
                const KaldiSparVector& labels_row1 = labels.Row(indx.t);
                int32_t label = labels_row1.GetElement(0).first;
                if (labels_row1.NumElements() > 1) {
                    float max_value = labels_row1.GetElement(0).second;
                    for (int32_t k = 1; k < labels_row1.NumElements(); ++k) {
                        const std::pair<MatrixIndexT, BaseFloat>& item = labels_row1.GetElement(k);
                        if (max_value < item.second) {
                            label = item.first;
                            max_value = item.second;
                        }
                    }
                }
                if (last_label != label) {
                    class_iter = frame_map.find(label);
                    if (class_iter == frame_map.end()) {
                        class_iter = frame_map.insert(std::make_pair(label, frame_arr_t())).first;
                    }
                    last_label = label;
                }
                const frame_arr_t& frame_arr = class_iter->second;
                const auto frame_indx = (size_t)(int_distrib(rand_gen) % frame_arr.size());
                const frame_pair_t& frame_pair = frame_arr.at(frame_indx);
                const KaldiSubVector& frame_row2 = frame_pair.first;
                const KaldiSparVector& labels_row2 = *frame_pair.second;
                const float scale2 = scale_distrib();
                const float scale1 = 1.0f - scale2;
                frame_row1.Scale(scale1);
                frame_row1.AddVec(scale2, frame_row2);
                typedef std::pair<int32_t, float> element_t;
                typedef std::map<int32_t, float> elements_t;
                elements_t elements;
                for (int32_t k = 0; k < labels_row1.NumElements(); ++k) {
                    const element_t& label = labels_row1.GetElement(k);
                    elements.insert(std::make_pair(label.first, scale1 * label.second));
                }
                for (int32_t k = 0; k < labels_row2.NumElements(); ++k) {
                    const element_t& label = labels_row2.GetElement(k);
                    auto iter = elements.find(label.first);
                    if (iter == elements.end()) {
                        elements.insert(std::make_pair(label.first, scale2 * label.second));
                    } else {
                        iter->second += scale2 * label.second;
                    }
                }
                KaldiSparVector labls_mixt(labels.NumCols(), std::vector<element_t>(elements.begin(), elements.end()));
                labels.SetRow(indx.t, labls_mixt);
                scale_count += scale2;
                adnum_count += 1;
            }
            gen_matrix = matrix;
            if (compress) {
                gen_matrix.Compress();
            }
            gen_labels = labels;
            ++num_mixed;
        } else {
            ++num_untouched;
        }
        const ExamplePair& example = pair;
        example_writer.Write(example.first, *example.second);
        ++num_wrote;
    }
}

void ExampleMixer::ShiftAndMixup(ExamplePair& _example) {
    GeneralMatrix& features = FindFeatures("input", _example.second->io);
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
    GeneralMatrix& labels = FindFeatures("output", _example.second->io);
    KaldiSparMatrix labels_main(labels.GetSparseMatrix());
    labels_main.Scale(exam_scale);
    KaldiSparMatrix labels_admx(labels.GetSparseMatrix());
    labels_admx.Scale(admx_scale);
    for (int32_t i = 0; i < labels_main.NumRows(); ++i) {
        int32_t admx_indx = i + shift;
        if (admx_indx < 0) {
            admx_indx = 0;
        } else if (admx_indx >= labels_main.NumRows()) {
            admx_indx = labels_main.NumRows() - 1;
        }
        typedef std::pair<int32_t, float> label_t;
        typedef std::map<int32_t, float> labels_t;
        labels_t lab_set;
        const KaldiSparVector& row_main = labels_main.Row(i);
        for (int32_t j = 0; j < row_main.NumElements(); ++j) {
            lab_set.insert(row_main.GetElement(j));
        }
        const KaldiSparVector& row_admx = labels_admx.Row(admx_indx);
        for (int32_t j = 0; j < row_admx.NumElements(); ++j) {
            const label_t& label = row_admx.GetElement(j);
            auto iter = lab_set.find(label.first);
            if (iter == lab_set.end()) {
                lab_set.insert(label);
            } else {
                iter->second += label.second;
            }
        }
        KaldiSparVector labls_dest(labels_main.NumCols(), std::vector<label_t>(lab_set.begin(), lab_set.end()));
        labels_main.SetRow(i, labls_dest);
    }
    labels = labels_main;
    scale_count += admx_scale;
    adnum_count += 1;
    shift_count += shift;
}

void ExampleMixer::AcceptExample(ExamplePair& _example) {
    if (mix_mode == "local") {
        const bool mixup = (float_distrib(rand_gen) > fixed_egs);
        if (mixup) {
            MixupLocal(_example);
            ++num_mixed;
        } else {
            ++num_untouched;
        }
        example_writer.Write(_example.first, *_example.second);
        ++num_wrote;
    } else if (mix_mode == "global") {
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
    } else if (mix_mode == "class") {
        if (egs_buffer.empty()) {
            egs_buffer.reserve(buff_size);
        }
        egs_buffer.push_back(_example);
        if (egs_buffer.size() == buff_size) {
            FlushClass(egs_buffer);
            egs_buffer.clear();
        }
    } else if (mix_mode == "shift") {
        const bool mixup = (float_distrib(rand_gen) > fixed_egs);
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
    } else if (mix_mode == "class") {
        FlushClass(egs_buffer);
    }
}

} }

bool AsBool(const char* _value) {
    if (_value == nullptr) {
        KALDI_ERR << "Pointer to bool string is nullptr.";
    }
    std::string value(_value);
    boost::to_lower(value);
    return (value == "true") || (value == "yes") || (value == "on") || (value == "1");
}

// --mix_mode=local ark:/media/work/coding/data/mgb3/train_data/egs.479.ark ark:/dev/null
// --mix_mode=global ark:/media/work/coding/data/mgb3/train_data/egs.479.ark ark:/dev/null
// --test_mode=1 --mix_mode=global ark:/media/work/coding/data/mgb3/train_data/cegs.10.ark ark:/dev/null
// --test_mode=1 --mix_mode=global --distrib=beta2:1.0 ark:/media/work/coding/data/mgb3/train_data/egs.479.ark ark:/dev/null

// --test_mode=1 --mix_mode=local --left_range=3 --right_range=3 ark:/mnt/TOSHIBA/khokhlov/coding/temp/i-vect/temp/mixup/egs.100.ark ark,t:/mnt/TOSHIBA/khokhlov/coding/temp/i-vect/temp/mixup/egs.100.mix.ark
// --test_mode=1 --mix_mode=local --left_range=3 --right_range=3 ark:/mnt/TOSHIBA/khokhlov/coding/temp/i-vect/temp/mixup/egs.101.ark ark,t:/mnt/TOSHIBA/khokhlov/coding/temp/i-vect/temp/mixup/egs.101.mix.ark

// --test_mode=1 --mix_mode=global ark:/mnt/TOSHIBA/khokhlov/coding/temp/i-vect/temp/mixup/egs.100.ark ark,t:/mnt/TOSHIBA/khokhlov/coding/temp/i-vect/temp/mixup/egs.100.mix.ark
// --test_mode=1 --mix_mode=global ark:/mnt/TOSHIBA/khokhlov/coding/temp/i-vect/temp/mixup/egs.101.ark ark,t:/mnt/TOSHIBA/khokhlov/coding/temp/i-vect/temp/mixup/egs.101.mix.ark

// --test_mode=1 --mix_mode=global --max_num=1 --distrib=uniform:0.1,0.7 ark:/mnt/TOSHIBA/khokhlov/coding/temp/i-vect/temp/mixup/egs.100.ark ark,t:/mnt/TOSHIBA/khokhlov/coding/temp/i-vect/temp/mixup/egs.100.mix.ark
// --test_mode=1 --mix_mode=global --max_num=1 --distrib=uniform:0.1,0.7 ark:/mnt/TOSHIBA/khokhlov/coding/temp/i-vect/temp/mixup/egs.101.ark ark,t:/mnt/TOSHIBA/khokhlov/coding/temp/i-vect/temp/mixup/egs.101.mix.ark
// --test_mode=1 --mix_mode=global --max_num=1 --distrib=beta:0.5        ark:/mnt/TOSHIBA/khokhlov/coding/temp/i-vect/temp/mixup/egs.101.ark ark,t:/mnt/TOSHIBA/khokhlov/coding/temp/i-vect/temp/mixup/egs.101.mix.ark
// --test_mode=1 --mix_mode=global --transform=sigmoid:10 ark:/mnt/TOSHIBA/khokhlov/coding/temp/i-vect/temp/mixup/egs.100.ark ark,t:/mnt/TOSHIBA/khokhlov/coding/temp/i-vect/temp/mixup/egs.100.mix.ark

// --test_mode=1 --mix_mode=local --max_num=1 ark:/mnt/TOSHIBA/khokhlov/coding/temp/i-vect/temp/mixup/egs.100.ark ark,t:/mnt/TOSHIBA/khokhlov/coding/temp/i-vect/temp/mixup/egs.100.mix.ark
// --test_mode=1 --mix_mode=local --max_num=1 ark:/mnt/TOSHIBA/khokhlov/coding/temp/i-vect/temp/mixup/egs.101.ark ark,t:/mnt/TOSHIBA/khokhlov/coding/temp/i-vect/temp/mixup/egs.101.mix.ark

// --test_mode=1 --mix_mode=class --max_num=1 --distrib=uniform:0.1,0.7 ark:/mnt/TOSHIBA/khokhlov/coding/temp/i-vect/temp/mixup/egs.100.ark ark,t:/mnt/TOSHIBA/khokhlov/coding/temp/i-vect/temp/mixup/egs.100.mix.ark
// --test_mode=1 --mix_mode=class --max_num=1 --distrib=uniform:0.1,0.7 ark:/mnt/TOSHIBA/khokhlov/coding/temp/i-vect/temp/mixup/egs.101.ark ark,t:/mnt/TOSHIBA/khokhlov/coding/temp/i-vect/temp/mixup/egs.101.mix.ark

// --test_mode=1 --mix_mode=shift --min-shift=2 --max-shift=5 --distrib=uniform:0.1,0.7 ark:/mnt/TOSHIBA/khokhlov/coding/temp/i-vect/temp/mixup/egs.100.ark ark,t:/mnt/TOSHIBA/khokhlov/coding/temp/i-vect/temp/mixup/egs.100.mix.ark

/*

work dir: /mnt/diskD/khokhlov/temp/mixup
ark:egs.112.ark ark:/dev/null
--mix-ivect=false ark:egs.112.ark ark:/dev/null
--mix-feats=false ark:egs.112.ark ark:/dev/null
--mix-labels=true ark:egs.112.ark ark:/dev/null
--mix-labels=false ark:egs.112.ark ark:/dev/null
--labels-map=1:2 ark:egs.112.ark ark:/dev/null

*/

int main(int argc, char *argv[]) {
    try {
        using namespace kaldi;
        using namespace kaldi::nnet3;

        const char *usage =
            "Usage:  nnet3-mixup-egs [options] <egs-rspecifier> <egs-wspecifier>\n"
            "\n"
            "e.g.\n"
            "nnet3-mixup-egs ark:train.egs ark:mixup.egs\n";
        ParseOptions po(usage);

        std::string mix_mode("global");
        const char* env_var = getenv("MIXUP_MIX_MODE");
        if (env_var != nullptr) {
            mix_mode = env_var;
        }
        po.Register("mix-mode", &mix_mode, R"(Mixup mode ("local", "global", "class", "shift") MIXUP_MIX_MODE)");

        std::string distrib("uniform:0.0,0.5");
        env_var = getenv("MIXUP_DISTRIB");
        if (env_var != nullptr) {
            distrib = env_var;
        }
        po.Register("distrib", &distrib, R"(Mixup scaling factors distribution ("uniform:min,max", "beta:alpha", "beta2:alpha") MIXUP_DISTRIB)");

        std::string transform;
        env_var = getenv("MIXUP_TRANSFORM");
        if (env_var != nullptr) {
            transform = env_var;
        }
        po.Register("transform", &transform, "Mixup scaling factor transform function for labels (\"sigmoid:k\") MIXUP_TRANSFORM");

        int32_t min_num = 1;
        env_var = getenv("MIXUP_MIN_NUM");
        if (env_var != nullptr) {
            min_num = boost::lexical_cast<int32_t>(env_var);
        }
        po.Register("min-num", &min_num, "Minimum number of admixtures MIXUP_MIN_NUM");

        int32_t max_num = 1;
        env_var = getenv("MIXUP_MAX_NUM");
        if (env_var != nullptr) {
            max_num = boost::lexical_cast<int32_t>(env_var);
        }
        po.Register("max-num", &max_num, "Maximum number of admixtures MIXUP_MAX_NUM");

        int32_t min_shift = 1;
        env_var = getenv("MIXUP_MIN_SHIFT");
        if (env_var != nullptr) {
            min_shift = boost::lexical_cast<int32_t>(env_var);
        }
        po.Register("min-shift", &min_shift, "Minimum sequence shift size (shift mode) MIXUP_MIN_SHIFT");

        int32_t max_shift = 3;
        env_var = getenv("MIXUP_MAX_SHIFT");
        if (env_var != nullptr) {
            max_shift = boost::lexical_cast<int32_t>(env_var);
        }
        po.Register("max-shift", &max_shift, "Maximum sequence shift size (shift mode) MIXUP_MAX_SHIFT");

        float fixed_egs = 0.10;
        env_var = getenv("MIXUP_FIXED_EGS");
        if (env_var != nullptr) {
            fixed_egs = boost::lexical_cast<float>(env_var);
        }
        po.Register("fixed-egs", &fixed_egs, "Portion of examples to leave untouched MIXUP_FIXED_EGS");

        float fixed_frames = 0.10;
        env_var = getenv("MIXUP_FIXED_FRAMES");
        if (env_var != nullptr) {
            fixed_frames = boost::lexical_cast<float>(env_var);
        }
        po.Register("fixed-frames", &fixed_frames, "Portion of frames to leave untouched MIXUP_FIXED_FRAMES");

        int32_t left_range = 3;
        env_var = getenv("MIXUP_LEFT_RANGE");
        if (env_var != nullptr) {
            left_range = boost::lexical_cast<int32_t>(env_var);
        }
        po.Register("left-range", &left_range, "Left range to pick an admixture frame (local mode) MIXUP_LEFT_RANGE");

        int32_t right_range = 3;
        env_var = getenv("MIXUP_RIGHT_RANGE");
        if (env_var != nullptr) {
            right_range = boost::lexical_cast<int32_t>(env_var);
        }
        po.Register("right-range", &right_range, "Right range to pick an admixture frame (local mode) MIXUP_RIGHT_RANGE");

        int32_t buff_size = 500;
        env_var = getenv("MIXUP_BUFF_SIZE");
        if (env_var != nullptr) {
            buff_size = boost::lexical_cast<int32_t>(env_var);
        }
        po.Register("buff-size", &buff_size, "Buffer size for data shuffling (global mode) MIXUP_BUFF_SIZE");

        bool mix_ivect = true;
        env_var = getenv("MIXUP_MIX_IVECT");
        if (env_var != nullptr) {
            mix_ivect = AsBool(env_var);
        }
        po.Register("mix-ivect", &mix_ivect, "Make i-vectors mixtures (MIXUP_MIX_IVECT)");

        bool mix_feats = true;
        env_var = getenv("MIXUP_MIX_FEATS");
        if (env_var != nullptr) {
            mix_feats = AsBool(env_var);
        }
        po.Register("mix-feats", &mix_feats, "Make features mixtures (MIXUP_MIX_FEATS)");

        bool mix_labels = true;
        env_var = getenv("MIXUP_MIX_LABELS");
        if (env_var != nullptr) {
            mix_labels = AsBool(env_var);
        }
        po.Register("mix-labels", &mix_labels, "Make labels mixtures (MIXUP_MIX_LABELS)");

        std::string labels_map_str;
        env_var = getenv("MIXUP_LABELS_MAP");
        if (env_var != nullptr) {
            labels_map_str = env_var;
        }
        po.Register("labels-map", &labels_map_str, "Map to transform labels in admixtures examples (MIXUP_LABELS_MAP)");

        bool compress = false;
        env_var = getenv("MIXUP_COMPRESS");
        if (env_var != nullptr) {
            compress = AsBool(env_var);
        }
        po.Register("compress", &compress, "Compress features and i-vectors MIXUP_COMPRESS");

        bool test_mode = false;
        po.Register("test-mode", &test_mode, "Self testing mode");

        po.Read(argc, argv);
        if (po.NumArgs() < 2) {
            po.PrintUsage();
            exit(1);
        }
        if (min_num < 1) {
            KALDI_ERR << "min_num must be greater or equal 1";
        }
        if (min_num > max_num) {
            KALDI_ERR << "min_num must be less or equal max_num";
        }
        if (mix_mode == "shift") {
            if (min_shift < 1) {
                KALDI_ERR << "min_shift must be greater or equal 1";
            }
            if (min_shift > max_shift) {
                KALDI_ERR << "min_shift must be less or equal max_shift";
            }
        }
        LabelsMap labels_map(labels_map_str);

        std::string examples_rspecifier = po.GetArg(1);
        std::string examples_wspecifier = po.GetArg(2);

        SequentialNnetExampleReader example_reader(examples_rspecifier);
        NnetExampleWriter example_writer(examples_wspecifier);

        ExampleMixer mixer(
            mix_mode, distrib, transform, example_writer, (size_t) min_num, (size_t) max_num,
            min_shift, max_shift, fixed_egs, fixed_frames, (size_t) left_range, (size_t) right_range,
            (size_t) buff_size, mix_ivect, mix_feats, mix_labels, labels_map, compress, test_mode
        );
        size_t num_read = 0;
        for (; !example_reader.Done(); example_reader.Next(), num_read++) {
            const NnetExample& example = example_reader.Value();
            ExamplePair ex_pair(example_reader.Key(), ExamplePtr(new NnetExample(example)));
            mixer.AcceptExample(ex_pair);
        }
        mixer.Finish();

        const FloatCounter& scale_count = mixer.ScaleCount();
        const IntCounter& shift_count = mixer.ShiftCount();
        const IntCounter& adnum_count = mixer.AdmixNumCount();
        const IntCounter& left_count = mixer.LeftCount();
        const IntCounter& right_count = mixer.RightCount();
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
        if (mix_mode == "global") {
            KALDI_LOG << "Average num: " << (boost::format("%.4f ( min: %d, max: %d )") % adnum_count.Average() % adnum_count.Minimum() % adnum_count.Maximum()).str();
        }
        if (mix_mode == "shift") {
            KALDI_LOG << "Average shift: " << (boost::format("%.4f ( min: %.4f, max: %.4f )") % shift_count.Average() % shift_count.Minimum() % shift_count.Maximum()).str();
        }
        if (mix_mode == "local") {
            KALDI_LOG << "Average total shift: " << (boost::format("%.4f ( min: %d, max: %d, num: %d )") % shift_count.Average() % shift_count.Minimum() % shift_count.Maximum() % shift_count.Count()).str();
            KALDI_LOG << "Average left shift: " << (boost::format("%.4f ( min: %d, max: %d, num: %d )") % left_count.Average() % left_count.Minimum() % left_count.Maximum() % left_count.Count()).str();
            KALDI_LOG << "Average right shift: " << (boost::format("%.4f ( min: %d, max: %d, num: %d )") % right_count.Average() % right_count.Minimum() % right_count.Maximum() % right_count.Count()).str();
        }

        return (num_wrote == 0 ? 1 : 0);
    } catch(const std::exception &e) {
        std::cerr << e.what() << '\n';
        return -1;
    }
}
