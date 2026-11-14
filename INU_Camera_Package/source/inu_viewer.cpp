// inu_viewer.cpp — Inuitive NU4000 彩色 / 深度实时预览
// 自包含 Win32 + GDI+ 界面，仅依赖 InuDev SDK 的 InuStreams.lib。

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <windowsx.h>
#include <objidl.h>
#include <gdiplus.h>
#include <wincodec.h>
#include <wrl/client.h>
#include <dwmapi.h>

#include "InuSensor.h"
#include "ImageStream.h"
#include "DepthStream.h"

#include <algorithm>
#include <cstring>
#include <objbase.h>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <ctime>
#include <memory>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#pragma comment(lib, "gdiplus.lib")
#pragma comment(lib, "windowscodecs.lib")
#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "dwmapi.lib")

using Microsoft::WRL::ComPtr;
using namespace Gdiplus;

// ---------------------------------------------------------------------------
// 常量 / 主题
// ---------------------------------------------------------------------------
#define WM_APP_FRAME (WM_APP + 1)
#define TIMER_TOAST  1

static const wchar_t* kClassName = L"InuColorDepthViewer";
static const wchar_t* kTitle     = L"INU 彩色 / 深度预览";

static const Color kBg        (255, 11, 13, 16);
static const Color kCard      (255, 18, 22, 29);
static const Color kCardEdge  (255, 40, 47, 61);
static const Color kText      (255, 232, 236, 244);
static const Color kMuted     (255, 140, 149, 169);
static const Color kFaint     (255, 94, 103, 122);
static const Color kAccent    (255, 79, 209, 197);
static const Color kWarn      (255, 255, 180, 84);
static const Color kError     (255, 255, 107, 107);
static const Color kChipBg    (150, 8, 10, 14);
static const Color kLetterbox (255, 12, 15, 20);

// ---------------------------------------------------------------------------
// 帧缓冲
// ---------------------------------------------------------------------------
struct ColorBuffer
{
    int w = 0, h = 0;
    uint32_t index = 0;
    std::vector<uint8_t> px;   // BGRA
};

struct DepthBuffer
{
    int w = 0, h = 0;
    uint32_t index = 0;
    std::vector<uint16_t> px;  // mm
};

static std::mutex g_frameMutex;
static std::shared_ptr<ColorBuffer> g_latestColor;
static std::shared_ptr<DepthBuffer> g_latestDepth;
static std::vector<std::shared_ptr<ColorBuffer>> g_colorPool(4);
static std::vector<std::shared_ptr<DepthBuffer>> g_depthPool(4);

// 界面线程展示中的帧（暂停时保持不变）
static std::shared_ptr<ColorBuffer> g_shownColor;
static std::shared_ptr<DepthBuffer> g_shownDepth;

// ---------------------------------------------------------------------------
// 状态
// ---------------------------------------------------------------------------
enum class ConnState { Connecting, Live, Failed };

static std::mutex g_statusMutex;
static ConnState g_conn = ConnState::Connecting;
static std::wstring g_connDetail = L"正在连接 INU 传感器…";
static double g_fps = 0.0;

static void SetStatus(ConnState s, const std::wstring& detail)
{
    std::lock_guard<std::mutex> lk(g_statusMutex);
    g_conn = s;
    g_connDetail = detail;
}

static void SetFps(double f)
{
    std::lock_guard<std::mutex> lk(g_statusMutex);
    g_fps = f;
}

// ---------------------------------------------------------------------------
// 全局
// ---------------------------------------------------------------------------
static HWND g_hwnd = nullptr;
static std::atomic<bool> g_running{ true };
static std::atomic<bool> g_workerDone{ false };
static std::thread g_worker;
static ULONG_PTR g_gdiplusToken = 0;
static ComPtr<IWICImagingFactory> g_wic;

struct Settings
{
    bool flip = true;        // 相机倒装，默认 180° 翻转
    bool autoRange = true;
    bool paused = false;
    int  colormap = 0;       // 0 turbo 1 viridis 2 inferno 3 gray
    int  minMm = 100;
    int  maxMm = 4000;
};
static Settings g_set;

static POINT g_mouse{ -1, -1 };
static bool  g_mouseInWindow = false;
static bool  g_trackingMouse = false;
static std::wstring g_toast;
static ULONGLONG g_toastUntil = 0;

// 深度卡片在屏幕上的位置，用于鼠标取深度
struct DepthView { RectF dest; int srcW = 0, srcH = 0; bool valid = false; };
static DepthView g_depthView;

static float DpiScale(HWND hwnd)
{
    UINT dpi = 96;
    typedef UINT (WINAPI* GetDpiForWindowFn)(HWND);
    static GetDpiForWindowFn fn = (GetDpiForWindowFn)GetProcAddress(
        GetModuleHandleW(L"user32.dll"), "GetDpiForWindow");
    if (fn) dpi = fn(hwnd);
    return (float)dpi / 96.0f;
}

static std::wstring Widen(const std::string& s)
{
    if (s.empty()) return L"";
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), nullptr, 0);
    std::wstring out((size_t)n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), &out[0], n);
    return out;
}

static std::wstring ExeDir()
{
    wchar_t buf[MAX_PATH];
    DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
    std::wstring p(buf, n);
    size_t pos = p.find_last_of(L"\\/");
    return pos == std::wstring::npos ? L"." : p.substr(0, pos);
}

// ---------------------------------------------------------------------------
// 配色表
// ---------------------------------------------------------------------------
static uint8_t g_lut[4][256][3];
static bool g_lutReady = false;

static float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

static void BuildLut()
{
    if (g_lutReady) return;
    for (int i = 0; i < 256; ++i)
    {
        float t = i / 255.0f;

        // 0: Turbo (Google polynomial)
        {
            float r = 0.13572138f + 4.61539260f*t - 42.66032258f*t*t + 132.13108234f*t*t*t
                    - 152.94239396f*t*t*t*t + 59.28637943f*t*t*t*t*t;
            float g = 0.09140261f + 2.19418839f*t + 4.84296658f*t*t - 14.18503333f*t*t*t
                    + 4.27729857f*t*t*t*t + 2.82956604f*t*t*t*t*t;
            float b = 0.10667330f + 12.64194608f*t - 60.58204836f*t*t + 110.36276771f*t*t*t
                    - 89.90310912f*t*t*t*t + 27.34824973f*t*t*t*t*t;
            g_lut[0][i][0] = (uint8_t)(clampf(r, 0, 1) * 255.0f);
            g_lut[0][i][1] = (uint8_t)(clampf(g, 0, 1) * 255.0f);
            g_lut[0][i][2] = (uint8_t)(clampf(b, 0, 1) * 255.0f);
        }

        struct Stop { float t; float r, g, b; };
        static const Stop viridis[] = {
            {0.00f, 68,   1,  84}, {0.25f, 59,  82, 139}, {0.50f, 33, 145, 140},
            {0.75f, 94, 201,  98}, {1.00f, 253, 231,  37},
        };
        static const Stop inferno[] = {
            {0.00f,   0,   0,   4}, {0.25f,  87,  16, 110}, {0.50f, 187,  55,  84},
            {0.75f, 249, 142,   9}, {1.00f, 252, 255, 164},
        };
        struct Lut { const Stop* s; int n; int slot; };
        const Lut maps[2] = { { viridis, 5, 1 }, { inferno, 5, 2 } };
        for (int m = 0; m < 2; ++m)
        {
            const Stop* st = maps[m].s;
            int n = maps[m].n;
            float r = st[n-1].r, g = st[n-1].g, b = st[n-1].b;
            for (int k = 0; k < n - 1; ++k)
            {
                if (t >= st[k].t && t <= st[k+1].t)
                {
                    float u = (t - st[k].t) / std::max(1e-6f, st[k+1].t - st[k].t);
                    r = st[k].r + (st[k+1].r - st[k].r) * u;
                    g = st[k].g + (st[k+1].g - st[k].g) * u;
                    b = st[k].b + (st[k+1].b - st[k].b) * u;
                    break;
                }
            }
            g_lut[maps[m].slot][i][0] = (uint8_t)clampf(r, 0, 255);
            g_lut[maps[m].slot][i][1] = (uint8_t)clampf(g, 0, 255);
            g_lut[maps[m].slot][i][2] = (uint8_t)clampf(b, 0, 255);
        }

        // 3: 灰度
        uint8_t v = (uint8_t)i;
        g_lut[3][i][0] = v; g_lut[3][i][1] = v; g_lut[3][i][2] = v;
    }
    g_lutReady = true;
}

static const wchar_t* ColormapName(int m)
{
    static const wchar_t* names[4] = { L"Turbo", L"Viridis", L"Inferno", L"灰度" };
    return names[m & 3];
}

// ---------------------------------------------------------------------------
// 绘制辅助
// ---------------------------------------------------------------------------
static void AddRoundRect(GraphicsPath& p, const RectF& r, REAL rad)
{
    REAL d = rad * 2.0f;
    if (d > r.Width)  d = r.Width;
    if (d > r.Height) d = r.Height;
    if (d <= 0) { p.AddRectangle(r); return; }
    p.AddArc(r.X, r.Y, d, d, 180.0f, 90.0f);
    p.AddArc(r.X + r.Width - d, r.Y, d, d, 270.0f, 90.0f);
    p.AddArc(r.X + r.Width - d, r.Y + r.Height - d, d, d, 0.0f, 90.0f);
    p.AddArc(r.X, r.Y + r.Height - d, d, d, 90.0f, 90.0f);
    p.CloseFigure();
}

static void FillRound(Graphics& g, const RectF& r, REAL rad, const Color& c)
{
    GraphicsPath p;
    AddRoundRect(p, r, rad);
    SolidBrush b(c);
    g.FillPath(&b, &p);
}

static void StrokeRound(Graphics& g, const RectF& r, REAL rad, const Color& c, REAL width)
{
    GraphicsPath p;
    AddRoundRect(p, r, rad);
    Pen pen(c, width);
    g.DrawPath(&pen, &p);
}

static void DrawTextG(Graphics& g, const std::wstring& s, Font& font, const Color& c,
                     float x, float y)
{
    SolidBrush b(c);
    StringFormat fmt;
    fmt.SetFormatFlags(StringFormatFlagsNoWrap | StringFormatFlagsMeasureTrailingSpaces);
    fmt.SetTrimming(StringTrimmingNone);
    g.DrawString(s.c_str(), -1, &font, PointF(x, y), &fmt, &b);
}

static float MeasureWidth(Graphics& g, const std::wstring& s, Font& font)
{
    RectF box;
    StringFormat fmt;
    fmt.SetFormatFlags(StringFormatFlagsNoWrap | StringFormatFlagsMeasureTrailingSpaces);
    g.MeasureString(s.c_str(), -1, &font, PointF(0, 0), &fmt, &box);
    return box.Width;
}

// 绘制一个小胶囊标签，返回宽度
static float DrawChip(Graphics& g, float x, float y, const std::wstring& text, Font& font,
                      const Color& bg, const Color& fg, float s)
{
    float padX = 9.0f * s, padY = 5.0f * s;
    float w = MeasureWidth(g, text, font) + padX * 2.0f;
    float h = 24.0f * s;
    FillRound(g, RectF(x, y, w, h), 7.0f * s, bg);
    DrawTextG(g, text, font, fg, x + padX, y + padY + 0.5f * s);
    return w;
}

// ---------------------------------------------------------------------------
// 图像
// ---------------------------------------------------------------------------
static std::vector<uint8_t> g_scratchColor;
static std::vector<uint8_t> g_scratchDepth;

static RectF FitRect(const RectF& box, int w, int h)
{
    if (w <= 0 || h <= 0) return box;
    float aspect = (float)w / (float)h;
    float dw = box.Width, dh = box.Width / aspect;
    if (dh > box.Height) { dh = box.Height; dw = box.Height * aspect; }
    return RectF(box.X + (box.Width - dw) * 0.5f, box.Y + (box.Height - dh) * 0.5f, dw, dh);
}

static void ClipRound(HDC dc, const RectF& r, REAL rad)
{
    HRGN rgn = CreateRoundRectRgn((int)std::floor(r.X), (int)std::floor(r.Y),
                                  (int)std::ceil(r.X + r.Width) + 1,
                                  (int)std::ceil(r.Y + r.Height) + 1,
                                  (int)(rad * 2.0f), (int)(rad * 2.0f));
    SelectClipRgn(dc, rgn);
    DeleteObject(rgn);
}

static void BlitColor(HDC dc, const ColorBuffer& f, const RectF& dest, bool flip)
{
    const int w = f.w, h = f.h;
    const uint8_t* data = f.px.data();
    if (flip)
    {
        g_scratchColor.resize((size_t)w * h * 4);
        for (int y = 0; y < h; ++y)
        {
            const uint32_t* src = reinterpret_cast<const uint32_t*>(f.px.data()) + (size_t)(h - 1 - y) * w;
            uint32_t* dst = reinterpret_cast<uint32_t*>(g_scratchColor.data()) + (size_t)y * w;
            for (int x = 0; x < w; ++x) dst[x] = src[w - 1 - x];
        }
        data = g_scratchColor.data();
    }
    BITMAPINFO bmi{};
    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = w;
    bmi.bmiHeader.biHeight = -h;
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 32;
    bmi.bmiHeader.biCompression = BI_RGB;
    StretchDIBits(dc, (int)dest.X, (int)dest.Y, (int)dest.Width, (int)dest.Height,
                  0, 0, w, h, data, &bmi, DIB_RGB_COLORS, SRCCOPY);
}

static void BlitDepth(HDC dc, const DepthBuffer& f, const RectF& dest, bool flip)
{
    BuildLut();
    const int w = f.w, h = f.h;
    const int map = g_set.colormap & 3;
    const int lo = std::min(g_set.minMm, g_set.maxMm - 1);
    const int hi = std::max(g_set.maxMm, lo + 1);
    const float k = 255.0f / (float)(hi - lo);

    g_scratchDepth.resize((size_t)w * h * 4);
    const uint8_t invalid[4] = { 18, 22, 29, 255 };

    for (int y = 0; y < h; ++y)
    {
        int sy = flip ? (h - 1 - y) : y;
        const uint16_t* src = f.px.data() + (size_t)sy * w;
        uint8_t* dst = g_scratchDepth.data() + (size_t)y * w * 4;
        for (int x = 0; x < w; ++x)
        {
            int sx = flip ? (w - 1 - x) : x;
            uint16_t v = src[sx];
            const uint8_t* c;
            if (v == 0 || v >= 65000) c = invalid;
            else
            {
                int t = (int)((v - lo) * k);
                if (t < 0) t = 0; else if (t > 255) t = 255;
                c = g_lut[map][t];
            }
            dst[0] = c[2]; dst[1] = c[1]; dst[2] = c[0]; dst[3] = 255;
            dst += 4;
        }
    }

    BITMAPINFO bmi{};
    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = w;
    bmi.bmiHeader.biHeight = -h;
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 32;
    bmi.bmiHeader.biCompression = BI_RGB;
    StretchDIBits(dc, (int)dest.X, (int)dest.Y, (int)dest.Width, (int)dest.Height,
                  0, 0, w, h, g_scratchDepth.data(), &bmi, DIB_RGB_COLORS, SRCCOPY);
}

static float CenterDistance(const DepthBuffer& f)
{
    int cx = f.w / 2, cy = f.h / 2;
    std::vector<uint16_t> vals;
    vals.reserve(441);
    for (int y = cy - 10; y <= cy + 10; ++y)
        for (int x = cx - 10; x <= cx + 10; ++x)
        {
            uint16_t v = f.px[(size_t)y * f.w + x];
            if (v > 0 && v < 65000) vals.push_back(v);
        }
    if (vals.empty()) return 0.0f;
    std::nth_element(vals.begin(), vals.begin() + vals.size() / 2, vals.end());
    return (float)vals[vals.size() / 2];
}

// 自动量程：直方图取 99 分位
static void UpdateAutoRange(const DepthBuffer& f)
{
    if (!g_set.autoRange) return;
    const int bins = 8192;
    static std::vector<uint32_t> hist(bins);
    std::fill(hist.begin(), hist.end(), 0);
    uint32_t total = 0;
    for (uint16_t v : f.px)
    {
        if (v > 0 && v < 65000)
        {
            int b = v / 2;
            if (b >= bins) b = bins - 1;
            ++hist[b];
            ++total;
        }
    }
    if (total < 1000) return;
    uint32_t target = (uint32_t)(total * 0.99);
    uint32_t acc = 0;
    int p99 = g_set.maxMm;
    for (int b = 0; b < bins; ++b)
    {
        acc += hist[b];
        if (acc >= target) { p99 = b * 2; break; }
    }
    int targetMm = std::max(g_set.minMm + 500, p99);
    // 平滑
    g_set.maxMm = (int)(g_set.maxMm * 0.8 + targetMm * 0.2);
}

// ---------------------------------------------------------------------------
// 保存
// ---------------------------------------------------------------------------
static bool EncodePngGrayscale16(const std::wstring& path, int w, int h, const std::vector<uint16_t>& px)
{
    if (!g_wic) return false;
    ComPtr<IWICStream> stream;
    if (FAILED(g_wic->CreateStream(&stream))) return false;
    if (FAILED(stream->InitializeFromFilename(path.c_str(), GENERIC_WRITE))) return false;

    ComPtr<IWICBitmapEncoder> enc;
    if (FAILED(g_wic->CreateEncoder(GUID_ContainerFormatPng, nullptr, &enc))) return false;
    if (FAILED(enc->Initialize(stream.Get(), WICBitmapEncoderNoCache))) return false;

    ComPtr<IWICBitmapFrameEncode> frame;
    if (FAILED(enc->CreateNewFrame(&frame, nullptr))) return false;
    if (FAILED(frame->Initialize(nullptr))) return false;
    if (FAILED(frame->SetSize((UINT)w, (UINT)h))) return false;

    WICPixelFormatGUID fmt = GUID_WICPixelFormat16bppGray;
    if (FAILED(frame->SetPixelFormat(&fmt))) return false;
    if (!IsEqualGUID(fmt, GUID_WICPixelFormat16bppGray)) return false;

    const UINT stride = (UINT)w * 2;
    if (FAILED(frame->WritePixels((UINT)h, stride, stride * (UINT)h,
                                  const_cast<BYTE*>(reinterpret_cast<const BYTE*>(px.data())))))
        return false;
    if (FAILED(frame->Commit())) return false;
    return SUCCEEDED(enc->Commit());
}

static bool EncodePngBgr24(const std::wstring& path, int w, int h, const std::vector<uint8_t>& bgr)
{
    if (!g_wic) return false;
    ComPtr<IWICStream> stream;
    if (FAILED(g_wic->CreateStream(&stream))) return false;
    if (FAILED(stream->InitializeFromFilename(path.c_str(), GENERIC_WRITE))) return false;

    ComPtr<IWICBitmapEncoder> enc;
    if (FAILED(g_wic->CreateEncoder(GUID_ContainerFormatPng, nullptr, &enc))) return false;
    if (FAILED(enc->Initialize(stream.Get(), WICBitmapEncoderNoCache))) return false;

    ComPtr<IWICBitmapFrameEncode> frame;
    if (FAILED(enc->CreateNewFrame(&frame, nullptr))) return false;
    if (FAILED(frame->Initialize(nullptr))) return false;
    if (FAILED(frame->SetSize((UINT)w, (UINT)h))) return false;

    WICPixelFormatGUID fmt = GUID_WICPixelFormat24bppBGR;
    if (FAILED(frame->SetPixelFormat(&fmt))) return false;
    if (!IsEqualGUID(fmt, GUID_WICPixelFormat24bppBGR)) return false;

    const UINT stride = (UINT)w * 3;
    if (FAILED(frame->WritePixels((UINT)h, stride, stride * (UINT)h,
                                  const_cast<BYTE*>(bgr.data()))))
        return false;
    if (FAILED(frame->Commit())) return false;
    return SUCCEEDED(enc->Commit());
}

static void ShowToast(const std::wstring& text)
{
    g_toast = text;
    g_toastUntil = GetTickCount64() + 2800;
    if (g_hwnd) SetTimer(g_hwnd, TIMER_TOAST, 120, nullptr);
    if (g_hwnd) InvalidateRect(g_hwnd, nullptr, FALSE);
}

static void SaveCapture()
{
    if (!g_shownColor || !g_shownDepth)
    {
        ShowToast(L"暂无图像可保存");
        return;
    }
    const ColorBuffer& c = *g_shownColor;
    const DepthBuffer& d = *g_shownDepth;
    const bool flip = g_set.flip;

    SYSTEMTIME st;
    GetLocalTime(&st);
    wchar_t base[96];
    swprintf(base, 96, L"%04d%02d%02d_%02d%02d%02d",
             st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond);

    std::wstring dir = ExeDir() + L"\\captures";
    CreateDirectoryW(dir.c_str(), nullptr);

    // 彩色 -> 24bpp BGR
    std::vector<uint8_t> bgr((size_t)c.w * c.h * 3);
    for (int y = 0; y < c.h; ++y)
    {
        int sy = flip ? (c.h - 1 - y) : y;
        const uint8_t* src = c.px.data() + (size_t)sy * c.w * 4;
        uint8_t* dst = bgr.data() + (size_t)y * c.w * 3;
        for (int x = 0; x < c.w; ++x)
        {
            int sx = flip ? (c.w - 1 - x) : x;
            const uint8_t* p = src + (size_t)sx * 4;
            dst[0] = p[0]; dst[1] = p[1]; dst[2] = p[2];
            dst += 3;
        }
    }

    std::vector<uint16_t> depth((size_t)d.w * d.h);
    for (int y = 0; y < d.h; ++y)
    {
        int sy = flip ? (d.h - 1 - y) : y;
        const uint16_t* src = d.px.data() + (size_t)sy * d.w;
        uint16_t* dst = depth.data() + (size_t)y * d.w;
        for (int x = 0; x < d.w; ++x)
            dst[x] = src[flip ? (d.w - 1 - x) : x];
    }

    std::wstring colorPath = dir + L"\\" + base + L"_color.png";
    std::wstring depthPath = dir + L"\\" + base + L"_depth.png";

    bool ok1 = EncodePngBgr24(colorPath, c.w, c.h, bgr);
    bool ok2 = EncodePngGrayscale16(depthPath, d.w, d.h, depth);

    if (ok1 && ok2) ShowToast(std::wstring(L"已保存  ") + base + L"_color.png / _depth.png");
    else ShowToast(L"保存失败（编码器不可用）");
}

// ---------------------------------------------------------------------------
// 传感器工作线程
// ---------------------------------------------------------------------------
static std::shared_ptr<ColorBuffer> AcquireColorBuffer(int w, int h)
{
    for (auto& b : g_colorPool)
    {
        if (!b) b = std::make_shared<ColorBuffer>();
        if (b.use_count() == 1)
        {
            b->w = w; b->h = h;
            b->px.resize((size_t)w * h * 4);
            return b;
        }
    }
    return nullptr;
}

static std::shared_ptr<DepthBuffer> AcquireDepthBuffer(int w, int h)
{
    for (auto& b : g_depthPool)
    {
        if (!b) b = std::make_shared<DepthBuffer>();
        if (b.use_count() == 1)
        {
            b->w = w; b->h = h;
            b->px.resize((size_t)w * h);
            return b;
        }
    }
    return nullptr;
}

static void PublishColor(const InuDev::CImageFrame& f)
{
    const int w = (int)f.Width();
    const int h = (int)f.Height();
    if (w <= 0 || h <= 0) return;
    const size_t size = f.GetDataSize();
    if (size < (size_t)w * h * 4) return;

    auto buf = AcquireColorBuffer(w, h);
    if (!buf) return;

    const size_t stride = size / (size_t)h;
    const uint8_t* src = f.GetData();
    for (int y = 0; y < h; ++y)
        memcpy(buf->px.data() + (size_t)y * w * 4, src + (size_t)y * stride, (size_t)w * 4);
    buf->index = f.FrameIndex;

    std::lock_guard<std::mutex> lk(g_frameMutex);
    g_latestColor = buf;
}

static void PublishDepth(const InuDev::CImageFrame& f)
{
    const int w = (int)f.Width();
    const int h = (int)f.Height();
    if (w <= 0 || h <= 0) return;
    const size_t size = f.GetDataSize();
    if (size < (size_t)w * h * 2) return;

    auto buf = AcquireDepthBuffer(w, h);
    if (!buf) return;

    const size_t stride = size / (size_t)h;
    const uint8_t* src = f.GetData();
    for (int y = 0; y < h; ++y)
        memcpy(buf->px.data() + (size_t)y * w, src + (size_t)y * stride, (size_t)w * 2);
    buf->index = f.FrameIndex;

    std::lock_guard<std::mutex> lk(g_frameMutex);
    g_latestDepth = buf;
}

static std::atomic<bool> g_restartRequested{ false };
static std::atomic<bool> g_hiRes{ true };

static bool RunSession()
{
    using namespace InuDev;

    auto fail = [](const std::wstring& msg)
    {
        SetStatus(ConnState::Failed, msg);
        if (g_hwnd) InvalidateRect(g_hwnd, nullptr, FALSE);
    };

    SetStatus(ConnState::Connecting, L"正在连接 INU 传感器…");

    std::shared_ptr<CInuSensor> sensor;
    try { sensor = CInuSensor::Create(); }
    catch (...) { sensor = nullptr; }
    if (!sensor) { fail(L"无法创建传感器对象"); return false; }

    CHwInformation hwInfo;
    CInuError rc = sensor->Init(hwInfo);
    if (rc != eOK) { fail(L"传感器初始化失败: " + Widen(std::string(rc))); return false; }
    if (!g_running) { sensor->Terminate(); return false; }

    // 选择分辨率档位并启动设备
    const bool hiRes = g_hiRes.load();
    std::map<uint32_t, CChannelSize> sizes;
    std::map<uint32_t, CChannelControlParams> params;
    for (auto& kv : hwInfo.GetChannels())
    {
        params[kv.first] = kv.second.ChannelControlParams;
        if (kv.second.ChannelType == eGeneralCameraChannel || kv.second.ChannelType == eDepthChannel)
        {
            params[kv.first].SensorRes = hiRes ? eFull : eBinning;
            params[kv.first].FPS = hiRes ? CChannelControlParams::USE_DEFAULT_FPS : 30;
        }
    }
    rc = sensor->Start(sizes, params);
    if (rc != eOK) { fail(L"传感器启动失败: " + Widen(std::string(rc))); sensor->Terminate(); return false; }

    auto color = sensor->CreateImageStream();
    auto depth = sensor->CreateDepthStream();
    if (!color || !depth) { fail(L"无法创建图像流"); sensor->Stop(); sensor->Terminate(); return false; }

    rc = color->Init(CImageStream::eBGRA);
    if (rc != eOK) { fail(L"彩色流初始化失败: " + Widen(std::string(rc))); sensor->Stop(); sensor->Terminate(); return false; }
    rc = color->Start();
    if (rc != eOK) { fail(L"彩色流启动失败: " + Widen(std::string(rc))); sensor->Stop(); sensor->Terminate(); return false; }

    rc = depth->Init(CDepthStream::eDepth);
    if (rc != eOK) { fail(L"深度流初始化失败: " + Widen(std::string(rc))); color->Stop(); color->Terminate(); sensor->Stop(); sensor->Terminate(); return false; }
    rc = depth->Start();
    if (rc != eOK) { fail(L"深度流启动失败: " + Widen(std::string(rc))); color->Stop(); color->Terminate(); sensor->Stop(); sensor->Terminate(); return false; }

    SetStatus(ConnState::Live, L"");
    if (g_hwnd) InvalidateRect(g_hwnd, nullptr, FALSE);

    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);
    auto now = [&]() { LARGE_INTEGER c; QueryPerformanceCounter(&c); return (double)c.QuadPart / (double)freq.QuadPart; };

    double t0 = now();
    int frames = 0;
    int consecutiveErrors = 0;

    while (g_running && !g_restartRequested)
    {
        std::shared_ptr<const CImageFrame> cf, df;
        CInuError ec = color->GetFrame(cf, 400);
        if (ec == eOK && cf && cf->Valid && cf->GetData()) PublishColor(*cf);

        CInuError ed = depth->GetFrame(df, 400);
        if (ed == eOK && df && df->Valid && df->GetData()) PublishDepth(*df);

        bool gotColor = (ec == eOK && cf && cf->Valid);
        bool gotDepth = (ed == eOK && df && df->Valid);
        if (gotColor || gotDepth) consecutiveErrors = 0;
        else if (ec != eTimeoutError && ed != eTimeoutError)
        {
            if (++consecutiveErrors == 5)
                fail(L"数据流中断: " + Widen(std::string(ec)) + L" / " + Widen(std::string(ed)));
        }

        frames++;
        double t1 = now();
        if (t1 - t0 >= 1.0)
        {
            SetFps(frames / (t1 - t0));
            frames = 0;
            t0 = t1;
        }

        if (g_hwnd) PostMessage(g_hwnd, WM_APP_FRAME, 0, 0);
    }

    color->Stop(); color->Terminate();
    depth->Stop(); depth->Terminate();
    sensor->Stop(); sensor->Terminate();
    return true;
}

static void WorkerMain()
{
    while (g_running)
    {
        g_restartRequested = false;
        bool ok = RunSession();
        if (!g_running) break;
        if (!ok) break;
        if (!g_restartRequested) break;
        {
            std::lock_guard<std::mutex> lk(g_statusMutex);
            g_conn = ConnState::Connecting;
            g_connDetail = L"正在切换分辨率…";
            g_fps = 0.0;
        }
        if (g_hwnd) InvalidateRect(g_hwnd, nullptr, FALSE);
    }
    SetStatus(ConnState::Failed, L"已停止");
    g_workerDone = true;
}

// ---------------------------------------------------------------------------
// 界面
// ---------------------------------------------------------------------------
static void PaintScene(HWND hwnd)
{
    PAINTSTRUCT ps;
    HDC hdc = BeginPaint(hwnd, &ps);
    RECT rc;
    GetClientRect(hwnd, &rc);
    const int W = rc.right, H = rc.bottom;
    if (W <= 0 || H <= 0) { EndPaint(hwnd, &ps); return; }

    const float s = DpiScale(hwnd);

    // 双缓冲
    static HDC mem = nullptr;
    static HBITMAP dib = nullptr;
    static HGDIOBJ oldBmp = nullptr;
    static int bw = 0, bh = 0;
    if (!mem || bw != W || bh != H)
    {
        if (mem) { SelectObject(mem, oldBmp); DeleteObject(dib); DeleteDC(mem); }
        mem = CreateCompatibleDC(hdc);
        BITMAPINFO bmi{};
        bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
        bmi.bmiHeader.biWidth = W;
        bmi.bmiHeader.biHeight = -H;
        bmi.bmiHeader.biPlanes = 1;
        bmi.bmiHeader.biBitCount = 32;
        bmi.bmiHeader.biCompression = BI_RGB;
        void* bits = nullptr;
        dib = CreateDIBSection(hdc, &bmi, DIB_RGB_COLORS, &bits, nullptr, 0);
        oldBmp = SelectObject(mem, dib);
        bw = W; bh = H;
    }

    // 取最新帧
    if (!g_set.paused)
    {
        std::lock_guard<std::mutex> lk(g_frameMutex);
        if (g_latestColor) g_shownColor = g_latestColor;
        if (g_latestDepth) g_shownDepth = g_latestDepth;
    }
    if (g_shownDepth) UpdateAutoRange(*g_shownDepth);

    {
        Graphics g(mem);
        g.SetSmoothingMode(SmoothingModeAntiAlias);
        g.SetTextRenderingHint(TextRenderingHintAntiAliasGridFit);
        g.SetPixelOffsetMode(PixelOffsetModeHalf);
        g.Clear(kBg);

        const float pad = 20.0f * s;
        const float gap = 16.0f * s;
        const float headerH = 74.0f * s;
        const float footerH = 54.0f * s;
        const float contentTop = headerH;
        const float contentH = (float)H - headerH - footerH;
        const float panelW = (W - pad * 2.0f - gap) * 0.5f;
        const float panelH = contentH - pad * 2.0f;
        const RectF cardL(pad, contentTop + pad, panelW, panelH);
        const RectF cardR(pad + panelW + gap, contentTop + pad, panelW, panelH);
        const REAL cardRad = 12.0f * s;

        // 字体
        Font fontTitle(L"Microsoft YaHei UI", 19.0f * s, FontStyleBold, UnitPixel);
        Font fontSub(L"Microsoft YaHei UI", 11.0f * s, FontStyleRegular, UnitPixel);
        Font fontChip(L"Microsoft YaHei UI", 10.5f * s, FontStyleRegular, UnitPixel);
        Font fontChipBold(L"Microsoft YaHei UI", 10.5f * s, FontStyleBold, UnitPixel);
        Font fontKey(L"Microsoft YaHei UI", 10.0f * s, FontStyleBold, UnitPixel);
        Font fontSmall(L"Microsoft YaHei UI", 10.5f * s, FontStyleRegular, UnitPixel);
        Font fontFps(L"Microsoft YaHei UI", 20.0f * s, FontStyleBold, UnitPixel);
        Font fontChipLg(L"Microsoft YaHei UI", 12.0f * s, FontStyleRegular, UnitPixel);

        ConnState conn;
        std::wstring detail;
        double fps;
        {
            std::lock_guard<std::mutex> lk(g_statusMutex);
            conn = g_conn; detail = g_connDetail; fps = g_fps;
        }

        // ---- 顶栏 ----
        SolidBrush accentBrush(kAccent);
        g.FillEllipse(&accentBrush, pad + 2.0f * s, 22.0f * s, 9.0f * s, 9.0f * s);
        DrawTextG(g, L"INU 实时预览", fontTitle, kText, pad + 20.0f * s, 13.0f * s);
        DrawTextG(g, L"Inuitive NU4000 · 彩色 + 深度", fontSub, kMuted, pad + 21.0f * s, 46.0f * s);

        // 状态胶囊（右侧）
        std::wstring statusText;
        Color statusColor;
        switch (conn)
        {
        case ConnState::Live:       statusText = L"已连接";   statusColor = kAccent; break;
        case ConnState::Connecting: statusText = L"连接中…";  statusColor = kWarn;   break;
        default:                    statusText = L"连接失败"; statusColor = kError;  break;
        }
        {
            float textW = MeasureWidth(g, statusText, fontChipBold);
            float pillW = textW + 34.0f * s;
            float pillH = 26.0f * s;
            float pillX = W - pad - pillW;
            float pillY = 22.0f * s;
            FillRound(g, RectF(pillX, pillY, pillW, pillH), 13.0f * s, Color(255, 22, 27, 36));
            StrokeRound(g, RectF(pillX, pillY, pillW, pillH), 13.0f * s, Color(255, 44, 52, 67), 1.0f);
            SolidBrush dot(statusColor);
            g.FillEllipse(&dot, pillX + 11.0f * s, pillY + 9.5f * s, 7.0f * s, 7.0f * s);
            DrawTextG(g, statusText, fontChipBold, kText, pillX + 24.0f * s, pillY + 5.0f * s);

            // FPS
            wchar_t fpsTxt[32];
            swprintf(fpsTxt, 32, L"%.1f", fps);
            float fw = MeasureWidth(g, fpsTxt, fontFps);
            DrawTextG(g, fpsTxt, fontFps, kText, pillX - 62.0f * s - fw, 15.0f * s);
            DrawTextG(g, L"FPS", fontSmall, kMuted, pillX - 56.0f * s, 27.0f * s);
        }

        // ---- 卡片 ----
        FillRound(g, cardL, cardRad, kCard);
        FillRound(g, cardR, cardRad, kCard);
        // 卡片内的留黑区域（等比缩放后两侧）
        RectF innerL(cardL.X + 1.0f, cardL.Y + 1.0f, cardL.Width - 2.0f, cardL.Height - 2.0f);
        RectF innerR(cardR.X + 1.0f, cardR.Y + 1.0f, cardR.Width - 2.0f, cardR.Height - 2.0f);
        SolidBrush letterbox(kLetterbox);
        GraphicsPath clipL, clipR;
        AddRoundRect(clipL, innerL, cardRad - 1.0f);
        AddRoundRect(clipR, innerR, cardRad - 1.0f);
        g.FillPath(&letterbox, &clipL);
        g.FillPath(&letterbox, &clipR);
        g.Flush(FlushIntentionSync);

        // ---- 图像（GDI 缩放） ----
        SetStretchBltMode(mem, HALFTONE);
        SetBrushOrgEx(mem, 0, 0, nullptr);

        if (g_shownColor)
        {
            RectF dst = FitRect(innerL, g_shownColor->w, g_shownColor->h);
            ClipRound(mem, innerL, cardRad - 1.0f);
            BlitColor(mem, *g_shownColor, dst, g_set.flip);
            SelectClipRgn(mem, nullptr);
        }
        else
        {
            g.FillPath(&letterbox, &clipL);
        }

        g_depthView.valid = false;
        if (g_shownDepth)
        {
            RectF dst = FitRect(innerR, g_shownDepth->w, g_shownDepth->h);
            ClipRound(mem, innerR, cardRad - 1.0f);
            BlitDepth(mem, *g_shownDepth, dst, g_set.flip);
            SelectClipRgn(mem, nullptr);
            g_depthView.dest = dst;
            g_depthView.srcW = g_shownDepth->w;
            g_depthView.srcH = g_shownDepth->h;
            g_depthView.valid = true;
        }
        else
        {
            g.FillPath(&letterbox, &clipR);
        }

        // ---- 覆盖层 ----
        g.SetSmoothingMode(SmoothingModeAntiAlias);

        // 彩色卡片标签
        {
            wchar_t buf[96];
            swprintf(buf, 96, L"彩色 COLOR   %d×%d",
                     g_shownColor ? g_shownColor->w : 0, g_shownColor ? g_shownColor->h : 0);
            DrawChip(g, cardL.X + 12.0f * s, cardL.Y + 12.0f * s, buf, fontChip, kChipBg, kText, s);
        }
        // 深度卡片标签
        {
            wchar_t buf[96];
            swprintf(buf, 96, L"深度 DEPTH   %d×%d",
                     g_shownDepth ? g_shownDepth->w : 0, g_shownDepth ? g_shownDepth->h : 0);
            float w1 = DrawChip(g, cardR.X + 12.0f * s, cardR.Y + 12.0f * s, buf, fontChip, kChipBg, kText, s);
            (void)w1;
            std::wstring cm = ColormapName(g_set.colormap);
            float cw = MeasureWidth(g, cm, fontChip);
            DrawChip(g, cardR.X + cardR.Width - 12.0f * s - (cw + 18.0f * s), cardR.Y + 12.0f * s,
                     cm, fontChip, kChipBg, kMuted, s);
        }

        // 深度：准星、中心距离、悬浮读数、色标
        if (g_shownDepth && g_depthView.valid)
        {
            const RectF& dst = g_depthView.dest;
            float ccx = dst.X + dst.Width * 0.5f;
            float ccy = dst.Y + dst.Height * 0.5f;
            Pen cross(Color(150, 255, 255, 255), 1.0f);
            g.DrawLine(&cross, ccx - 9.0f * s, ccy, ccx + 9.0f * s, ccy);
            g.DrawLine(&cross, ccx, ccy - 9.0f * s, ccx, ccy + 9.0f * s);

            float cd = CenterDistance(*g_shownDepth);
            wchar_t buf[64];
            swprintf(buf, 64, L"中心 %.2f m", cd / 1000.0f);
            DrawChip(g, dst.X + 12.0f * s, dst.Y + dst.Height - 36.0f * s, buf, fontChip,
                     kChipBg, kText, s);

            // 鼠标读数
            if (g_mouseInWindow && g_mouse.x >= dst.X && g_mouse.x < dst.X + dst.Width &&
                g_mouse.y >= dst.Y && g_mouse.y < dst.Y + dst.Height)
            {
                float u = (g_mouse.x - dst.X) / dst.Width;
                float v = (g_mouse.y - dst.Y) / dst.Height;
                int px = (int)(u * g_depthView.srcW);
                int py = (int)(v * g_depthView.srcH);
                if (g_set.flip) { px = g_depthView.srcW - 1 - px; py = g_depthView.srcH - 1 - py; }
                px = std::max(0, std::min(g_depthView.srcW - 1, px));
                py = std::max(0, std::min(g_depthView.srcH - 1, py));
                uint16_t dmm = g_shownDepth->px[(size_t)py * g_shownDepth->w + px];

                Pen hoverPen(kAccent, 1.0f);
                g.DrawEllipse(&hoverPen, (float)g_mouse.x - 6.0f * s, (float)g_mouse.y - 6.0f * s,
                              12.0f * s, 12.0f * s);

                if (dmm > 0 && dmm < 65000)
                    swprintf(buf, 64, L"%.2f m", dmm / 1000.0f);
                else
                    swprintf(buf, 64, L"无数据");
                float tw = MeasureWidth(g, buf, fontChipBold) + 18.0f * s;
                float tx = g_mouse.x + 14.0f * s;
                float ty = g_mouse.y - 30.0f * s;
                if (tx + tw > dst.X + dst.Width - 6.0f * s) tx = g_mouse.x - tw - 14.0f * s;
                if (ty < dst.Y + 6.0f * s) ty = g_mouse.y + 14.0f * s;
                DrawChip(g, tx, ty, buf, fontChipBold, Color(220, 0, 0, 0),
                         (dmm > 0 && dmm < 65000) ? kAccent : kMuted, s);
            }

            // 色标
            {
                const float barW = 10.0f * s;
                const float barH = std::min(220.0f * s, dst.Height - 96.0f * s);
                if (barH > 60.0f * s)
                {
                    float bx = dst.X + dst.Width - 18.0f * s;
                    float by = dst.Y + (dst.Height - barH) * 0.5f;
                    // 底色
                    FillRound(g, RectF(bx - 6.0f * s, by - 24.0f * s, barW + 12.0f * s, barH + 48.0f * s),
                              8.0f * s, Color(140, 0, 0, 0));
                    BuildLut();
                    const int map = g_set.colormap & 3;
                    for (int i = 0; i < 48; ++i)
                    {
                        float t0 = 1.0f - i / 48.0f;
                        int idx = (int)(t0 * 255.0f);
                        const uint8_t* c = g_lut[map][idx];
                        SolidBrush seg(Color(255, c[0], c[1], c[2]));
                        g.FillRectangle(&seg, bx, by + barH * i / 48.0f, barW, barH / 48.0f + 1.0f);
                    }

                    wchar_t b1[32], b2[32];
                    swprintf(b1, 32, L"%.1f m", g_set.maxMm / 1000.0f);
                    swprintf(b2, 32, L"%.1f m", g_set.minMm / 1000.0f);
                    DrawTextG(g, b1, fontSmall, kText, bx - 30.0f * s, by - 20.0f * s);
                    DrawTextG(g, b2, fontSmall, kText, bx - 30.0f * s, by + barH + 4.0f * s);
                }
            }
        }

        // 错误 / 占位提示
        if (conn != ConnState::Live)
        {
            std::wstring msg = (conn == ConnState::Connecting)
                ? (detail.empty() ? std::wstring(L"正在连接传感器…") : detail)
                : detail;
            Font fontMsg(L"Microsoft YaHei UI", 13.0f * s, FontStyleRegular, UnitPixel);
            RectF boxL = innerL, boxR = innerR;
            for (const RectF* box : { &boxL, &boxR })
            {
                float tw = std::min(MeasureWidth(g, msg, fontMsg), box->Width - 40.0f * s);
                (void)tw;
                StringFormat fmt;
                fmt.SetAlignment(StringAlignmentCenter);
                fmt.SetLineAlignment(StringAlignmentCenter);
                SolidBrush b(conn == ConnState::Failed ? kError : kMuted);
                g.DrawString(msg.c_str(), -1, &fontMsg, *box, &fmt, &b);
            }
        }

        // 暂停标记
        if (g_set.paused)
        {
            std::wstring t = L"已暂停";
            float tw = MeasureWidth(g, t, fontChipBold) + 26.0f * s;
            FillRound(g, RectF((W - tw) * 0.5f, contentTop + 8.0f * s, tw, 26.0f * s),
                      13.0f * s, Color(210, 30, 36, 46));
            DrawTextG(g, t, fontChipBold, kWarn, (W - tw) * 0.5f + 13.0f * s, contentTop + 13.0f * s);
        }

        // ---- 底栏 ----
        {
            float y = (float)H - footerH + 14.0f * s;
            float x = pad;
            struct Hint { const wchar_t* key; const wchar_t* label; };
            const Hint hints[] = {
                { L"S", L"保存" }, { L"A", L"自动量程" }, { L"M", L"配色" },
                { L"F", L"翻转" }, { L"R", L"清晰度" }, { L"P", L"暂停" }, { L"Esc", L"退出" },
            };
            for (const Hint& h : hints)
            {
                float kw = MeasureWidth(g, h.key, fontKey) + 14.0f * s;
                float kh = 20.0f * s;
                if (x + kw + 62.0f * s > W * 0.72f) break;
                // 键帽
                FillRound(g, RectF(x, y, kw, kh), 5.0f * s, Color(255, 26, 31, 41));
                StrokeRound(g, RectF(x, y, kw, kh), 5.0f * s, Color(255, 52, 60, 77), 1.0f);
                DrawTextG(g, h.key, fontKey, kText, x + 7.0f * s, y + 3.0f * s);
                DrawTextG(g, h.label, fontSmall, kMuted, x + kw + 6.0f * s, y + 3.5f * s);
                x += kw + 6.0f * s + MeasureWidth(g, h.label, fontSmall) + 18.0f * s;
            }

            // 右侧量程
            wchar_t rng[96];
            swprintf(rng, 96, L"量程  %.2f – %.2f m  ·  %s",
                     g_set.minMm / 1000.0f, g_set.maxMm / 1000.0f,
                     g_set.autoRange ? L"自动" : L"手动");
            float rw = MeasureWidth(g, rng, fontSmall);
            DrawTextG(g, rng, fontSmall, g_set.autoRange ? kAccent : kMuted,
                     W - pad - rw, y + 3.5f * s);
        }

        // Toast
        if (!g_toast.empty() && GetTickCount64() < g_toastUntil)
        {
            float tw = MeasureWidth(g, g_toast, fontChipLg) + 32.0f * s;
            float th = 36.0f * s;
            float tx = (W - tw) * 0.5f;
            float ty = (float)H - footerH - th - 18.0f * s;
            FillRound(g, RectF(tx, ty, tw, th), 18.0f * s, Color(235, 26, 32, 42));
            StrokeRound(g, RectF(tx, ty, tw, th), 18.0f * s, Color(255, 52, 62, 78), 1.0f);
            DrawTextG(g, g_toast, fontChipLg, kText, tx + 16.0f * s, ty + 9.0f * s);
        }

        g.Flush(FlushIntentionSync);
    }

    BitBlt(hdc, 0, 0, W, H, mem, 0, 0, SRCCOPY);
    EndPaint(hwnd, &ps);
}

// ---------------------------------------------------------------------------
// 窗口过程
// ---------------------------------------------------------------------------
static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp)
{
    switch (msg)
    {
    case WM_CREATE:
        g_hwnd = hwnd;
        {
            BOOL dark = TRUE;
            DwmSetWindowAttribute(hwnd, 20, &dark, sizeof(dark));
            DwmSetWindowAttribute(hwnd, 19, &dark, sizeof(dark));
        }
        break;

    case WM_ERASEBKGND:
        return 1;

    case WM_PAINT:
        PaintScene(hwnd);
        return 0;

    case WM_APP_FRAME:
        InvalidateRect(hwnd, nullptr, FALSE);
        return 0;

    case WM_TIMER:
        if (wp == TIMER_TOAST && GetTickCount64() >= g_toastUntil)
        {
            KillTimer(hwnd, TIMER_TOAST);
            g_toast.clear();
            InvalidateRect(hwnd, nullptr, FALSE);
        }
        return 0;

    case WM_MOUSEMOVE:
    {
        if (!g_trackingMouse)
        {
            TRACKMOUSEEVENT tme{ sizeof(tme), TME_LEAVE, hwnd, 0 };
            TrackMouseEvent(&tme);
            g_trackingMouse = true;
        }
        g_mouse.x = GET_X_LPARAM(lp);
        g_mouse.y = GET_Y_LPARAM(lp);
        g_mouseInWindow = true;
        InvalidateRect(hwnd, nullptr, FALSE);
        return 0;
    }

    case WM_MOUSELEAVE:
        g_mouseInWindow = false;
        g_trackingMouse = false;
        InvalidateRect(hwnd, nullptr, FALSE);
        return 0;

    case WM_SETCURSOR:
        if (LOWORD(lp) == HTCLIENT && g_depthView.valid)
        {
            POINT p;
            GetCursorPos(&p);
            ScreenToClient(hwnd, &p);
            if (p.x >= g_depthView.dest.X && p.x < g_depthView.dest.X + g_depthView.dest.Width &&
                p.y >= g_depthView.dest.Y && p.y < g_depthView.dest.Y + g_depthView.dest.Height)
            {
                SetCursor(LoadCursor(nullptr, IDC_CROSS));
                return TRUE;
            }
        }
        break;

    case WM_KEYDOWN:
        switch (wp)
        {
        case VK_ESCAPE:
        case 'Q':
            PostMessage(hwnd, WM_CLOSE, 0, 0);
            return 0;
        case 'S':
            SaveCapture();
            return 0;
        case 'A':
            g_set.autoRange = !g_set.autoRange;
            ShowToast(g_set.autoRange ? L"深度量程：自动" : L"深度量程：手动");
            return 0;
        case 'M':
            g_set.colormap = (g_set.colormap + 1) % 4;
            ShowToast(std::wstring(L"配色：") + ColormapName(g_set.colormap));
            return 0;
        case 'F':
            g_set.flip = !g_set.flip;
            ShowToast(g_set.flip ? L"画面翻转 180°：开" : L"画面翻转 180°：关");
            return 0;
        case 'R':
            g_hiRes = !g_hiRes;
            g_restartRequested = true;
            ShowToast(g_hiRes ? L"清晰度：全分辨率（较慢）" : L"清晰度：流畅（低分率高帧率）");
            return 0;
        case 'P':
        case VK_SPACE:
            g_set.paused = !g_set.paused;
            ShowToast(g_set.paused ? L"已暂停（按 P 继续）" : L"已继续");
            return 0;
        case VK_OEM_PLUS:
        case VK_ADD:
            g_set.maxMm = std::min(15000, g_set.maxMm + 250);
            g_set.autoRange = false;
            return 0;
        case VK_OEM_MINUS:
        case VK_SUBTRACT:
            g_set.maxMm = std::max(g_set.minMm + 500, g_set.maxMm - 250);
            g_set.autoRange = false;
            return 0;
        }
        break;

    case WM_GETMINMAXINFO:
        {
            MINMAXINFO* mmi = (MINMAXINFO*)lp;
            float s = DpiScale(hwnd);
            mmi->ptMinTrackSize.x = (LONG)(980 * s);
            mmi->ptMinTrackSize.y = (LONG)(620 * s);
        }
        return 0;

    case WM_DPICHANGED:
        {
            RECT* prc = (RECT*)lp;
            SetWindowPos(hwnd, nullptr, prc->left, prc->top,
                         prc->right - prc->left, prc->bottom - prc->top,
                         SWP_NOZORDER | SWP_NOACTIVATE);
            InvalidateRect(hwnd, nullptr, FALSE);
        }
        return 0;

    case WM_CLOSE:
        DestroyWindow(hwnd);
        return 0;

    case WM_DESTROY:
        g_running = false;
        if (g_worker.joinable())
        {
            // 最多等 5 秒让工作线程收尾（USB 断开时 SDK 清理可能卡住）
            for (int i = 0; i < 50 && !g_workerDone.load(); ++i)
                Sleep(100);
            if (g_workerDone.load()) g_worker.join();
            else g_worker.detach();
        }
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProc(hwnd, msg, wp, lp);
}

// ---------------------------------------------------------------------------
// 入口
// ---------------------------------------------------------------------------
static void EnableDpiAwareness()
{
    typedef BOOL (WINAPI* SetCtxFn)(HANDLE);
    HMODULE user32 = GetModuleHandleW(L"user32.dll");
    auto fn = (SetCtxFn)GetProcAddress(user32, "SetProcessDpiAwarenessContext");
    if (fn)
    {
        // DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 == (HANDLE)-4
        if (fn((HANDLE)-4)) return;
    }
    SetProcessDPIAware();
}

int WINAPI wWinMain(HINSTANCE hInst, HINSTANCE, LPWSTR, int)
{
    EnableDpiAwareness();
    CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);

    GdiplusStartupInput gsi;
    GdiplusStartup(&g_gdiplusToken, &gsi, nullptr);

    CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
                     IID_PPV_ARGS(&g_wic));

    WNDCLASSEXW wc{};
    wc.cbSize = sizeof(wc);
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInst;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground = nullptr;
    wc.lpszClassName = kClassName;
    RegisterClassExW(&wc);

    HDC screen = GetDC(nullptr);
    int sw = GetDeviceCaps(screen, LOGPIXELSX);
    ReleaseDC(nullptr, screen);
    float scale = sw / 96.0f;

    int winW = (int)(1320 * scale);
    int winH = (int)(820 * scale);

    RECT work;
    SystemParametersInfoW(SPI_GETWORKAREA, 0, &work, 0);
    int x = work.left + ((work.right - work.left) - winW) / 2;
    int y = work.top + ((work.bottom - work.top) - winH) / 2;

    HWND hwnd = CreateWindowExW(
        0, kClassName, kTitle,
        WS_OVERLAPPEDWINDOW,
        x, y, winW, winH,
        nullptr, nullptr, hInst, nullptr);
    if (!hwnd) return 1;

    ShowWindow(hwnd, SW_SHOW);
    UpdateWindow(hwnd);

    g_worker = std::thread(WorkerMain);

    MSG msg;
    while (GetMessage(&msg, nullptr, 0, 0) > 0)
    {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    if (g_worker.joinable()) g_worker.join();

    g_wic.Reset();
    GdiplusShutdown(g_gdiplusToken);
    CoUninitialize();
    return 0;
}
