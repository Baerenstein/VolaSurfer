# VolaSurfer WebGL Wallpaper

A stunning 3D animated desktop background powered by real-time options volatility data from VolaSurfer.

## 🌟 Features

- **Real-time 3D Volatility Surface**: Beautiful WebGL-rendered volatility surface with live data updates
- **Interactive Controls**: Mouse and keyboard controls for camera movement and zoom
- **Live Statistics**: Real-time display of spread, volume, and market statistics
- **Responsive Design**: Works on any screen resolution
- **Smooth Animations**: 60fps animations with Three.js
- **WebSocket Support**: Optional real-time data streaming

## 🚀 Quick Start

### Prerequisites

1. **VolaSurfer Backend**: Make sure the VolaSurfer backend is running on `http://localhost:8000`
2. **Modern Browser**: Chrome, Firefox, Safari, or Edge with WebGL support
3. **Python 3.7+**: For the local development server

### Installation

1. **Start the VolaSurfer Backend**:
   ```bash
   cd backend
   python -m uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Start the WebGL Wallpaper Server**:
   ```bash
   cd webgl-wallpaper
   python server.py
   ```

3. **Open in Browser**:
   Navigate to `http://localhost:8080`

## 🎮 Controls

### Mouse Controls
- **Left Click + Drag**: Rotate camera around the scene
- **Mouse Wheel**: Zoom in/out
- **Right Click + Drag**: Pan camera (if implemented)

### UI Controls
- **Pause/Play Button**: Toggle animation
- **Reset View Button**: Reset camera to default position

## 📊 Data Visualization

### 3D Volatility Surface
- **X-axis**: Moneyness (strike price relative to current price)
- **Y-axis**: Days to Expiry
- **Z-axis**: Implied Volatility
- **Color**: Green gradient representing volatility levels

### Trend Line
- **Orange Line**: Historical average volatility over time
- **Position**: Floats above the main surface

### Real-time Statistics
- **Spread**: Maximum minus minimum volatility (percentage)
- **Volume**: Number of data points in the surface
- **Last Update**: Timestamp of the most recent data

## 🔧 Configuration

### API Endpoints

The wallpaper connects to these VolaSurfer API endpoints:

- `GET /api/surface_snapshot` - 3D surface data (x, y, z coordinates)
- `GET /api/trendline` - Historical trend data
- `GET /api/stats` - Live statistics
- `WS /api/stream` - WebSocket real-time updates (optional)

### Customization

Edit `main.js` to customize:

- **Update Interval**: Change `updateInterval` (default: 5000ms)
- **Animation Speed**: Modify rotation speeds in the `animate()` function
- **Colors**: Update material colors in `createSurfaceMesh()` and `createTrendline()`
- **Camera Position**: Adjust initial camera position in `setupThreeJS()`

## 🖥️ Desktop Wallpaper Setup

### Ubuntu/GNOME

1. **Install GNOME Wallpaper Extension**:
   ```bash
   sudo apt install gnome-shell-extensions
   ```

2. **Use as Desktop Background**:
   - Open GNOME Extensions
   - Install "Desktop Wallpaper" extension
   - Set the wallpaper URL to `http://localhost:8080`

### Alternative: Chromium Kiosk Mode

1. **Create a startup script**:
   ```bash
   #!/bin/bash
   chromium-browser --kiosk --app=http://localhost:8080
   ```

2. **Set as startup application**:
   - Add to System Settings > Startup Applications
   - Set to run on login

### macOS

1. **Use Wallpaper Engine** (if available)
2. **Or create a simple app wrapper** using Electron

## 🛠️ Development

### Project Structure

```
webgl-wallpaper/
├── index.html          # Main HTML file
├── main.js            # Three.js WebGL logic
├── style.css          # UI styling
├── server.py          # Local development server
└── README.md          # This file
```

### Adding New Features

1. **New Data Sources**: Add API endpoints in `backend/server/app.py`
2. **Visual Effects**: Extend Three.js scene in `main.js`
3. **UI Elements**: Add HTML elements and style them in `style.css`

### Performance Optimization

- **Reduce Geometry Complexity**: Lower the plane geometry segments in `createSurfaceMesh()`
- **Optimize Updates**: Increase `updateInterval` for less frequent updates
- **Disable Shadows**: Set `renderer.shadowMap.enabled = false` for better performance

## 🐛 Troubleshooting

### Common Issues

1. **"Failed to load market data"**
   - Ensure VolaSurfer backend is running on port 8000
   - Check browser console for CORS errors
   - Verify API endpoints are accessible

2. **Poor Performance**
   - Reduce geometry complexity
   - Disable shadows
   - Lower update frequency

3. **WebGL Not Supported**
   - Update graphics drivers
   - Try a different browser
   - Check WebGL support at `chrome://gpu`

4. **Port Already in Use**
   - Change port in `server.py`
   - Kill existing processes: `lsof -ti:8080 | xargs kill`

### Debug Mode

Enable debug logging by adding this to `main.js`:

```javascript
// Add at the top of the class
this.debug = true;

// Add debug logging throughout the code
if (this.debug) console.log('Debug message');
```

## 📈 Future Enhancements

- [ ] **Particle Effects**: Add floating particles around the surface
- [ ] **Sound Integration**: Audio feedback for market events
- [ ] **Multiple Assets**: Support for different cryptocurrencies
- [ ] **Advanced Shaders**: Custom GLSL shaders for better visuals
- [ ] **Mobile Support**: Touch controls for mobile devices
- [ ] **Export Features**: Screenshot and video recording
- [ ] **Themes**: Multiple color schemes and visual styles

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is part of VolaSurfer and follows the same license terms.

## 🙏 Acknowledgments

- **Three.js**: 3D graphics library
- **VolaSurfer**: Options volatility analysis platform
- **FastAPI**: Modern Python web framework

---

**Enjoy your beautiful, data-driven desktop wallpaper! 🎨📊** 