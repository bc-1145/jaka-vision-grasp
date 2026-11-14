// Python bindings for the Inuitive InuDev 4.x C++ SDK (Windows x64).
//
// The official InuDev Python package (pythonsdk/) is only distributed by
// Inuitive support and is not part of the public 4.x installer, so this module
// wraps the C++ API that is actually installed on this machine.
//
// Build: pysdk_src\build_pysdk.bat  ->  ..\pythonsdk\_inudev<EXT>.pyd
//
// Low level API (see pythonsdk\api for the friendlier wrapper):
//   s = _inudev.Sensor(); s.init(); s.start(binning=False, fps=0)
//   img = s.create_image_stream(); img.init("BGRA"); img.start()
//   f = img.get_frame(2000)   # _inudev.Frame or None
//   buf = f.data             # bytes, BGRA, width*height*4

#define NOMINMAX

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <DepthStream.h>
#include <HwInformation.h>
#include <ImageStream.h>
#include <InuSensor.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>

namespace py = pybind11;
using namespace InuDev;

namespace {

std::string to_upper(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return s;
}

std::string err_text(const CInuError& err)
{
    std::string text = static_cast<std::string>(err);
    if (text.empty())
        text = "InuDev error " + std::to_string(static_cast<int>(err));
    return text;
}

void check(const CInuError& err, const char* what)
{
    if (static_cast<int>(err) != static_cast<int>(eOK))
        throw std::runtime_error(std::string(what) + " failed: " + err_text(err));
}

const char* channel_type_name(EChannelType type)
{
    switch (type) {
    case eGeneralCameraChannel: return "GENERAL_CAMERA";
    case eTrackingChannel:      return "TRACKING";
    case eStereoChannel:        return "STEREO";
    case eDepthChannel:         return "DEPTH";
    case eFeaturesTrackingChannel: return "FEATURES_TRACKING";
    default:                    return "UNKNOWN";
    }
}

const char* format_name(unsigned int fmt)
{
    switch (fmt) {
    case eEmpty: return "EMPTY";
    case eDepth: return "DEPTH";
    case eBGR:   return "BGR";
    case eBGRA:  return "BGRA";
    case eRGBA:  return "RGBA";
    case eRGB:   return "RGB";
    case eDisparity: return "DISPARITY";
    default:     return "UNKNOWN";
    }
}

/////////////////////////////////////////////////////////////////////////////
// Frames
/////////////////////////////////////////////////////////////////////////////

struct PyFrame
{
    std::shared_ptr<const CImageFrame> frame;

    PyFrame() = default;
    explicit PyFrame(std::shared_ptr<const CImageFrame> f) : frame(std::move(f)) {}

    const CImageFrame& ref() const
    {
        if (!frame)
            throw std::runtime_error("empty frame");
        return *frame;
    }
};

py::bytes frame_bytes(const PyFrame& pf)
{
    const CImageFrame& f = pf.ref();
    const InuDev::byte* data = f.GetData();
    const uint32_t size = f.GetDataSize();
    if (!data || size == 0)
        return py::bytes();
    return py::bytes(reinterpret_cast<const char*>(data), size);
}

/////////////////////////////////////////////////////////////////////////////
// Sensor
/////////////////////////////////////////////////////////////////////////////

class PySensor
{
public:
    std::shared_ptr<CInuSensor> sensor;
    CHwInformation hw;
    bool has_hw = false;
    bool started = false;
    std::shared_ptr<py::object> state_callback;

    PySensor() : sensor(CInuSensor::Create())
    {
        if (!sensor)
            throw std::runtime_error("InuDev: CInuSensor::Create() returned null");
        state_callback = std::make_shared<py::object>(py::none());
    }

    py::dict hw_dict() const
    {
        py::dict out;
        py::list channels;
        for (const auto& kv : hw.GetChannels()) {
            py::dict c;
            c["channel_id"] = kv.first;
            c["type"] = static_cast<int>(kv.second.ChannelType);
            c["type_name"] = channel_type_name(kv.second.ChannelType);
            c["sensor_ids"] = kv.second.ChannelSensorsID;
            channels.append(c);
        }
        out["channels"] = channels;

        py::list sensors;
        for (const auto& kv : hw.GetSensors()) {
            py::dict s;
            s["sensor_id"] = kv.first;
            s["model"] = static_cast<int>(kv.second.Model);
            s["role"] = static_cast<int>(kv.second.Role);
            s["lens_type"] = kv.second.LensType;
            s["channels"] = kv.second.ConnectedChannels;
            sensors.append(s);
        }
        out["sensors"] = sensors;
        return out;
    }

    py::dict init()
    {
        if (started)
            throw std::runtime_error("sensor is already started");
        if (!has_hw) {
            py::gil_scoped_release release;
            check(sensor->Init(hw), "CInuSensor::Init");
        }
        has_hw = true;
        return hw_dict();
    }

    py::dict start(bool binning, int fps)
    {
        if (!has_hw)
            throw std::runtime_error("init() must be called before start()");
        if (started)
            throw std::runtime_error("sensor is already started");

        std::map<uint32_t, CChannelSize> sizes;
        std::map<uint32_t, CChannelControlParams> params;
        for (const auto& kv : hw.GetChannels()) {
            params[kv.first] = kv.second.ChannelControlParams;
            if (kv.second.ChannelType == eGeneralCameraChannel ||
                kv.second.ChannelType == eDepthChannel) {
                params[kv.first].SensorRes = binning ? eBinning : eFull;
                params[kv.first].FPS = fps > 0
                    ? static_cast<uint32_t>(fps)
                    : static_cast<uint32_t>(CChannelControlParams::USE_DEFAULT_FPS);
            }
        }

        {
            py::gil_scoped_release release;
            check(sensor->Start(sizes, params), "CInuSensor::Start");
        }
        started = true;

        py::dict out;
        py::dict channels;
        for (const auto& kv : sizes) {
            py::list wh;
            wh.append(kv.second[0]);
            wh.append(kv.second[1]);
            channels[py::int_(kv.first)] = wh;
        }
        out["channels"] = channels;
        return out;
    }

    void stop()
    {
        if (sensor) {
            py::gil_scoped_release release;
            sensor->Stop();
        }
        started = false;
    }

    void terminate()
    {
        if (sensor) {
            py::gil_scoped_release release;
            sensor->Terminate();
        }
        started = false;
    }

    void close()
    {
        try { stop(); } catch (...) {}
        try { terminate(); } catch (...) {}
    }

    int connection_state() const { return static_cast<int>(sensor->GetConnectionState()); }
    int state() const { return static_cast<int>(sensor->GetState()); }

    void set_state_callback(py::object callback)
    {
        if (callback.is_none()) {
            *state_callback = py::none();
            py::gil_scoped_release release;
            sensor->Register(nullptr);
            return;
        }
        *state_callback = callback;
        auto holder = state_callback;
        py::gil_scoped_release release;
        sensor->Register([holder](std::shared_ptr<CInuSensor>, EConnectionState st, CInuError err) {
            py::gil_scoped_acquire gil;
            if (holder->is_none())
                return;
            try {
                (*holder)(py::int_(static_cast<int>(st)),
                          py::int_(static_cast<int>(err)),
                          py::str(err_text(err)));
            } catch (py::error_already_set& e) {
                e.restore();
            }
        });
    }
};

/////////////////////////////////////////////////////////////////////////////
// Image stream
/////////////////////////////////////////////////////////////////////////////

class PyImageStream
{
public:
    std::shared_ptr<CInuSensor> owner;
    std::shared_ptr<CImageStream> stream;

    PyImageStream(std::shared_ptr<CInuSensor> o, std::shared_ptr<CImageStream> s)
        : owner(std::move(o)), stream(std::move(s))
    {
        if (!stream)
            throw std::runtime_error("InuDev: CreateImageStream() returned null");
    }

    void init(const std::string& format)
    {
        const std::string f = to_upper(format);
        CImageStream::EOutputFormat out = CImageStream::eBGRA;
        if (f == "RGBA")        out = CImageStream::eRGBA;
        else if (f == "BGR")    out = CImageStream::eBGR;
        else if (f == "RAW")    out = CImageStream::eRaw;
        else if (f == "DEFAULT") out = CImageStream::eDefault;
        py::gil_scoped_release release;
        check(stream->Init(out), "CImageStream::Init");
    }

    void start()
    {
        py::gil_scoped_release release;
        check(stream->Start(), "CImageStream::Start");
    }
    void stop()
    {
        py::gil_scoped_release release;
        stream->Stop();
    }
    void terminate()
    {
        py::gil_scoped_release release;
        stream->Terminate();
    }
    int state() const { return static_cast<int>(stream->State()); }

    py::object get_frame(unsigned int timeout_ms)
    {
        std::shared_ptr<const CImageFrame> frame;
        CInuError err(eOK);
        {
            py::gil_scoped_release release;
            err = stream->GetFrame(frame, timeout_ms);
        }
        if (static_cast<int>(err) == static_cast<int>(eTimeoutError))
            return py::none();
        check(err, "CImageStream::GetFrame");
        if (!frame)
            return py::none();
        return py::cast(PyFrame(frame));
    }
};

/////////////////////////////////////////////////////////////////////////////
// Depth stream
/////////////////////////////////////////////////////////////////////////////

class PyDepthStream
{
public:
    std::shared_ptr<CInuSensor> owner;
    std::shared_ptr<CDepthStream> stream;

    PyDepthStream(std::shared_ptr<CInuSensor> o, std::shared_ptr<CDepthStream> s)
        : owner(std::move(o)), stream(std::move(s))
    {
        if (!stream)
            throw std::runtime_error("InuDev: CreateDepthStream() returned null");
    }

    void init(const std::string& format)
    {
        const std::string f = to_upper(format);
        CDepthStream::EOutputFormat out = CDepthStream::eDepth;
        if (f == "BGRA")        out = CDepthStream::eBGRA;
        else if (f == "RGBA")   out = CDepthStream::eRGBA;
        else if (f == "RGB")    out = CDepthStream::eRGB;
        else if (f == "RAW")    out = CDepthStream::eRaw;
        else if (f == "DEFAULT") out = CDepthStream::eDefault;
        py::gil_scoped_release release;
        check(stream->Init(out), "CDepthStream::Init");
    }

    void start()
    {
        py::gil_scoped_release release;
        check(stream->Start(), "CDepthStream::Start");
    }
    void stop()
    {
        py::gil_scoped_release release;
        stream->Stop();
    }
    void terminate()
    {
        py::gil_scoped_release release;
        stream->Terminate();
    }
    int state() const { return static_cast<int>(stream->State()); }

    py::object get_frame(unsigned int timeout_ms)
    {
        std::shared_ptr<const CImageFrame> frame;
        CInuError err(eOK);
        {
            py::gil_scoped_release release;
            err = stream->GetFrame(frame, timeout_ms);
        }
        if (static_cast<int>(err) == static_cast<int>(eTimeoutError))
            return py::none();
        check(err, "CDepthStream::GetFrame");
        if (!frame)
            return py::none();
        return py::cast(PyFrame(frame));
    }
};

} // namespace

/////////////////////////////////////////////////////////////////////////////
// Module
/////////////////////////////////////////////////////////////////////////////

PYBIND11_MODULE(_inudev, m)
{
    m.doc() = "Inuitive InuDev 4.x bindings (built from the installed C++ SDK)";
    m.attr("__version__") = "4.16";

    py::class_<PyFrame>(m, "Frame", "One image or depth frame.")
        .def_property_readonly("width", [](const PyFrame& f) { return f.ref().Width(); })
        .def_property_readonly("height", [](const PyFrame& f) { return f.ref().Height(); })
        .def_property_readonly("bytes_per_pixel", [](const PyFrame& f) { return f.ref().BytesPerPixel(); })
        .def_property_readonly("data_size", [](const PyFrame& f) { return f.ref().GetDataSize(); })
        .def_property_readonly("format_code", [](const PyFrame& f) { return f.ref().Format(); })
        .def_property_readonly("format_name", [](const PyFrame& f) { return format_name(f.ref().Format()); })
        .def_property_readonly("frame_index", [](const PyFrame& f) { return f.ref().FrameIndex; })
        .def_property_readonly("timestamp_ns", [](const PyFrame& f) { return f.ref().Timestamp; })
        .def_property_readonly("valid", [](const PyFrame& f) { return f.ref().Valid; })
        .def_property_readonly("data", &frame_bytes, "Frame buffer as bytes (BGRA / uint16 mm).")
        .def("tobytes", &frame_bytes)
        .def("__bytes__", &frame_bytes)
        .def("__len__", [](const PyFrame& f) { return f.ref().GetDataSize(); })
        .def("__repr__", [](const PyFrame& f) {
            const CImageFrame& r = f.ref();
            return "<_inudev.Frame " + std::to_string(r.Width()) + "x" +
                   std::to_string(r.Height()) + " " + format_name(r.Format()) + ">";
        });

    py::class_<PyImageStream>(m, "ImageStream")
        .def("init", &PyImageStream::init, py::arg("format") = "BGRA")
        .def("start", &PyImageStream::start)
        .def("stop", &PyImageStream::stop)
        .def("terminate", &PyImageStream::terminate)
        .def("get_frame", &PyImageStream::get_frame, py::arg("timeout_ms") = 2000)
        .def_property_readonly("state", &PyImageStream::state);

    py::class_<PyDepthStream>(m, "DepthStream")
        .def("init", &PyDepthStream::init, py::arg("format") = "DEPTH")
        .def("start", &PyDepthStream::start)
        .def("stop", &PyDepthStream::stop)
        .def("terminate", &PyDepthStream::terminate)
        .def("get_frame", &PyDepthStream::get_frame, py::arg("timeout_ms") = 2000)
        .def_property_readonly("state", &PyDepthStream::state);

    py::class_<PySensor>(m, "Sensor")
        .def(py::init<>())
        .def("init", &PySensor::init)
        .def("start", &PySensor::start, py::arg("binning") = false, py::arg("fps") = 0)
        .def("stop", &PySensor::stop)
        .def("terminate", &PySensor::terminate)
        .def("close", &PySensor::close)
        .def("set_state_callback", &PySensor::set_state_callback, py::arg("callback"))
        .def("create_image_stream", [](PySensor& s) {
            return PyImageStream(s.sensor, s.sensor->CreateImageStream());
        })
        .def("create_depth_stream", [](PySensor& s) {
            return PyDepthStream(s.sensor, s.sensor->CreateDepthStream());
        })
        .def_property_readonly("connection_state", &PySensor::connection_state)
        .def_property_readonly("state", &PySensor::state)
        .def_property_readonly("hw_info", &PySensor::hw_dict);
}
