class HistoricalSurfaceViewer {
    constructor() {
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.surfaceMesh = null;
        this.isAnimating = true;
        this.animationId = null;
        
        // Data management
        this.historicalData = [];
        this.currentSurfaceIndex = 0;
        this.assets = [];
        
        // Playback controls
        this.isPlaying = false;
        this.playbackInterval = null;
        this.playbackSpeed = 1000;
        
        // Settings
        this.interpolationDensity = 20;
        this.showWireframe = false;
        this.selectedAsset = null;
        this.dataLimit = 500;
        
        // API endpoints
        this.apiBase = 'http://localhost:8000';
        this.endpoints = {
            history: '/api/v1/vol_surface/history',
            assets: '/api/v1/assets'
        };
        
        this.init();
    }
    
    init() {
        this.setupThreeJS();
        this.setupEventListeners();
        this.updateConnectionStatus('Loading...');
        this.loadAssets();
        this.loadHistoricalData();
        this.animate();
    }
    
    setupThreeJS() {
        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x000000);
        
        // Camera
        this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        this.camera.position.set(8, 8, 8); // Zoomed in by 20% (from 10,10,10 to 8,8,8)
        this.camera.lookAt(0, 0, 0);
        
        // Renderer
        this.renderer = new THREE.WebGLRenderer({ 
            canvas: document.getElementById('webgl-canvas'),
            antialias: true 
        });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        
        // Lighting
        this.setupLighting();
        
        // Grid helper
        const gridHelper = new THREE.GridHelper(20, 20, 0x444444, 0x222222);
        this.scene.add(gridHelper);
        
        // Handle window resize
        window.addEventListener('resize', () => {
            this.camera.aspect = window.innerWidth / window.innerHeight;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(window.innerWidth, window.innerHeight);
        });
    }
    
    setupLighting() {
        // Ambient light
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
        this.scene.add(ambientLight);
        
        // Directional light
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(10, 10, 5);
        directionalLight.castShadow = true;
        this.scene.add(directionalLight);
        
        // Point light for dramatic effect
        const pointLight = new THREE.PointLight(0x00aaff, 0.5, 20);
        pointLight.position.set(0, 5, 0);
        this.scene.add(pointLight);
    }
    
    setupEventListeners() {
        // Mouse controls for camera
        let isMouseDown = false;
        let mouseX = 0;
        let mouseY = 0;
        
        document.addEventListener('mousedown', (event) => {
            isMouseDown = true;
            mouseX = event.clientX;
            mouseY = event.clientY;
        });
        
        document.addEventListener('mouseup', () => {
            isMouseDown = false;
        });
        
        document.addEventListener('mousemove', (event) => {
            if (isMouseDown) {
                const deltaX = event.clientX - mouseX;
                const deltaY = event.clientY - mouseY;
                
                // Rotate camera around the scene
                const spherical = new THREE.Spherical();
                spherical.setFromVector3(this.camera.position);
                spherical.theta -= deltaX * 0.01;
                spherical.phi += deltaY * 0.01;
                spherical.phi = Math.max(0.1, Math.min(Math.PI - 0.1, spherical.phi));
                
                this.camera.position.setFromSpherical(spherical);
                this.camera.lookAt(0, 0, 0);
                
                mouseX = event.clientX;
                mouseY = event.clientY;
            }
        });
        
        // Mouse wheel for zoom
        document.addEventListener('wheel', (event) => {
            const zoomSpeed = 0.1;
            const direction = event.deltaY > 0 ? 1 : -1;
            const distance = this.camera.position.length();
            const newDistance = Math.max(2, Math.min(20, distance + direction * zoomSpeed));
            
            this.camera.position.normalize().multiplyScalar(newDistance);
        });
        
        // Control event listeners
        document.getElementById('asset-select').addEventListener('change', (e) => {
            this.selectedAsset = e.target.value ? Number(e.target.value) : null;
            this.stopPlayback();
            this.loadHistoricalData();
        });
        
        document.getElementById('limit-select').addEventListener('change', (e) => {
            this.dataLimit = Number(e.target.value);
            this.stopPlayback();
            this.loadHistoricalData();
        });
        
        document.getElementById('interpolation-select').addEventListener('change', (e) => {
            this.interpolationDensity = Number(e.target.value);
            this.updateSurfaceVisualization();
        });
        
        document.getElementById('wireframe-toggle').addEventListener('change', (e) => {
            this.showWireframe = e.target.checked;
            this.updateSurfaceVisualization();
        });
        
        document.getElementById('refresh-btn').addEventListener('click', () => {
            this.loadHistoricalData();
        });
        
        // Playback controls
        document.getElementById('play-pause-btn').addEventListener('click', () => {
            this.togglePlayback();
        });
        
        document.getElementById('step-forward-btn').addEventListener('click', () => {
            this.stepForward();
        });
        
        document.getElementById('step-back-btn').addEventListener('click', () => {
            this.stepBackward();
        });
        
        document.getElementById('reset-btn').addEventListener('click', () => {
            this.resetToStart();
        });
        
        document.getElementById('end-btn').addEventListener('click', () => {
            this.goToEnd();
        });
        
        document.getElementById('timeline-slider').addEventListener('input', (e) => {
            this.stopPlayback();
            const value = Number(e.target.value);
            this.currentSurfaceIndex = this.historicalData.length - 1 - value;
            this.updateSurfaceVisualization();
            this.updateUI();
        });
        
        document.getElementById('speed-select').addEventListener('change', (e) => {
            this.playbackSpeed = Number(e.target.value);
            if (this.isPlaying) {
                this.startPlayback();
            }
        });
        
        document.getElementById('surface-select').addEventListener('change', (e) => {
            this.stopPlayback();
            this.currentSurfaceIndex = Number(e.target.value);
            this.updateSurfaceVisualization();
            this.updateUI();
        });
    }
    
    async loadAssets() {
        try {
            console.log('Loading assets from:', `${this.apiBase}${this.endpoints.assets}`);
            const response = await fetch(`${this.apiBase}${this.endpoints.assets}`);
            console.log('Assets response status:', response.status);
            
            if (response.ok) {
                this.assets = await response.json();
                console.log('Loaded assets:', this.assets);
                this.populateAssetSelect();
                this.updateConnectionStatus('Connected');
            } else {
                const errorText = await response.text();
                console.error('Assets API error:', response.status, errorText);
                this.updateConnectionStatus('Error');
            }
        } catch (error) {
            console.error('Error loading assets:', error);
            this.updateConnectionStatus('Error');
        }
    }
    
    populateAssetSelect() {
        const select = document.getElementById('asset-select');
        select.innerHTML = '<option value="">All Assets</option>';
        
        this.assets.forEach(asset => {
            const option = document.createElement('option');
            option.value = asset.id;
            option.textContent = `${asset.ticker} (${asset.asset_type})`;
            select.appendChild(option);
        });
    }
    
    async loadHistoricalData() {
        this.showLoading(true);
        
        try {
            const params = new URLSearchParams({
                limit: this.dataLimit.toString()
            });
            
            if (this.selectedAsset) {
                params.append('asset_id', this.selectedAsset.toString());
            }
            
            const url = `${this.apiBase}${this.endpoints.history}?${params}`;
            console.log('Loading historical data from:', url);
            
            const response = await fetch(url);
            console.log('Historical data response status:', response.status);
            
            if (response.ok) {
                this.historicalData = await response.json();
                console.log('Loaded historical data:', this.historicalData.length, 'surfaces');
                this.currentSurfaceIndex = this.historicalData.length - 1; // Start with oldest
                this.updateUI();
                this.updateSurfaceVisualization();
                this.showPlaybackControls();
                this.updateConnectionStatus('Connected');
            } else {
                const errorText = await response.text();
                console.error('Historical data API error:', response.status, errorText);
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
        } catch (error) {
            console.error('Error loading historical data:', error);
            this.showError('Failed to load historical data: ' + error.message);
            this.updateConnectionStatus('Error');
        } finally {
            this.showLoading(false);
        }
    }
    
    updateSurfaceVisualization() {
        if (!this.historicalData.length || this.currentSurfaceIndex < 0) {
            return;
        }
        
        const surface = this.historicalData[this.currentSurfaceIndex];
        this.createSurfaceMesh(surface);
    }
    
    createSurfaceMesh(surface) {
        console.log('Creating surface mesh for surface:', surface);
        
        // Remove existing surface mesh
        if (this.surfaceMesh) {
            this.scene.remove(this.surfaceMesh);
        }
        
        // Check for different possible property names
        const moneyness = surface.moneyness || surface.moneyness_array || [];
        const daysToExpiry = surface.days_to_expiry || surface.daysToExpiry || surface.dte || [];
        const impliedVols = surface.implied_vols || surface.impliedVols || surface.volatility || [];
        
        console.log('Extracted arrays:', {
            moneyness: moneyness.length,
            daysToExpiry: daysToExpiry.length,
            impliedVols: impliedVols.length,
            moneynessSample: moneyness.slice(0, 5),
            daysToExpirySample: daysToExpiry.slice(0, 5),
            impliedVolsSample: impliedVols.slice(0, 5)
        });
        
        if (!moneyness.length || !daysToExpiry.length || !impliedVols.length) {
            console.warn('Invalid surface data - missing required arrays');
            console.log('Surface data keys:', Object.keys(surface));
            return;
        }
        
        // Get unique values for grid dimensions
        const uniqueMoneyness = [...new Set(moneyness)].sort((a, b) => a - b);
        const uniqueDaysToExpiry = [...new Set(daysToExpiry)].sort((a, b) => a - b);
        
        console.log('Grid dimensions:', {
            uniqueMoneyness: uniqueMoneyness.length,
            uniqueDaysToExpiry: uniqueDaysToExpiry.length,
            interpolationDensity: this.interpolationDensity
        });
        
        if (uniqueMoneyness.length === 0 || uniqueDaysToExpiry.length === 0) {
            console.warn('No valid data points');
            return;
        }
        
        // Create interpolated grid
        const moneynessMin = Math.min(...uniqueMoneyness);
        const moneynessMax = Math.max(...uniqueMoneyness);
        const dteMin = Math.min(...uniqueDaysToExpiry);
        const dteMax = Math.max(...uniqueDaysToExpiry);
        
        const gridMoneyness = Array.from({length: this.interpolationDensity}, (_, i) => 
            moneynessMin + (moneynessMax - moneynessMin) * i / (this.interpolationDensity - 1)
        );
        const gridDTE = Array.from({length: this.interpolationDensity}, (_, i) => 
            dteMin + (dteMax - dteMin) * i / (this.interpolationDensity - 1)
        );
        
        // Create geometry
        const geometry = new THREE.PlaneGeometry(8, 8, this.interpolationDensity - 1, this.interpolationDensity - 1);
        const vertices = geometry.attributes.position.array;
        
        // Create color array for vertex colors
        const colors = new Float32Array(vertices.length);
        
        console.log('Created geometry with vertices:', vertices.length / 3);
        
        // Find min/max volatility for color scaling
        const allVols = impliedVols.filter(v => typeof v === 'number' && !isNaN(v));
        const minVol = Math.min(...allVols);
        const maxVol = Math.max(...allVols);
        
        // Interpolate values
        let validPoints = 0;
        for (let i = 0; i < this.interpolationDensity; i++) {
            for (let j = 0; j < this.interpolationDensity; j++) {
                const vertexIndex = (i * this.interpolationDensity + j) * 3;
                const gridX = gridMoneyness[j];
                const gridY = gridDTE[i];
                
                // Find interpolated volatility value
                let interpolatedValue = 0;
                let minDistance = Infinity;
                
                for (let k = 0; k < impliedVols.length; k++) {
                    const dataMoneyness = moneyness[k];
                    const dataDTE = daysToExpiry[k];
                    const dataVol = impliedVols[k];
                    
                    if (typeof dataVol === 'number' && !isNaN(dataVol)) {
                        const distance = Math.sqrt(
                            Math.pow((gridX - dataMoneyness) / (moneynessMax - moneynessMin), 2) +
                            Math.pow((gridY - dataDTE) / (dteMax - dteMin), 2)
                        );
                        
                        if (distance < minDistance) {
                            minDistance = distance;
                            interpolatedValue = dataVol / 100; // Convert percentage to decimal
                        }
                    }
                }
                
                // Apply height based on volatility
                const height = interpolatedValue * 4; // Scale for visibility
                vertices[vertexIndex + 2] = height;
                
                // Calculate Viridis color based on volatility
                const normalizedVol = (interpolatedValue * 100 - minVol) / (maxVol - minVol);
                const color = this.viridisColor(normalizedVol);
                
                colors[vertexIndex] = color.r;     // Red
                colors[vertexIndex + 1] = color.g; // Green
                colors[vertexIndex + 2] = color.b; // Blue
                
                if (height > 0) {
                    validPoints++;
                }
            }
        }
        
        // Set vertex colors
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        
        console.log('Interpolated surface with valid points:', validPoints);
        
        geometry.attributes.position.needsUpdate = true;
        geometry.computeVertexNormals();
        
        console.log('Geometry bounds:', geometry.boundingBox);
        
        // Create material with vertex colors for gradient
        const material = new THREE.MeshPhongMaterial({
            vertexColors: true,
            transparent: true,
            opacity: 0.8,
            side: THREE.DoubleSide,
            wireframe: this.showWireframe
        });
        
        // Create mesh
        this.surfaceMesh = new THREE.Mesh(geometry, material);
        this.surfaceMesh.rotation.x = -Math.PI / 2;
        this.surfaceMesh.rotation.z = Math.PI; // Flip X and Y axes
        this.surfaceMesh.castShadow = true;
        this.surfaceMesh.receiveShadow = true;
        
        console.log('Created surface mesh:', this.surfaceMesh);
        console.log('Mesh position:', this.surfaceMesh.position);
        console.log('Mesh rotation:', this.surfaceMesh.rotation);
        
        this.scene.add(this.surfaceMesh);
        console.log('Added mesh to scene. Scene children count:', this.scene.children.length);
    }
    
    // Playback controls
    togglePlayback() {
        if (this.isPlaying) {
            this.stopPlayback();
        } else {
            this.startPlayback();
        }
    }
    
    startPlayback() {
        if (this.historicalData.length <= 1) return;
        
        this.isPlaying = true;
        this.updatePlaybackButton();
        
        this.playbackInterval = setInterval(() => {
            this.currentSurfaceIndex--;
            if (this.currentSurfaceIndex < 0) {
                this.currentSurfaceIndex = this.historicalData.length - 1;
            }
            this.updateSurfaceVisualization();
            this.updateUI();
        }, this.playbackSpeed);
    }
    
    stopPlayback() {
        this.isPlaying = false;
        this.updatePlaybackButton();
        
        if (this.playbackInterval) {
            clearInterval(this.playbackInterval);
            this.playbackInterval = null;
        }
    }
    
    stepForward() {
        this.stopPlayback();
        this.currentSurfaceIndex = Math.max(0, this.currentSurfaceIndex - 1);
        this.updateSurfaceVisualization();
        this.updateUI();
    }
    
    stepBackward() {
        this.stopPlayback();
        this.currentSurfaceIndex = Math.min(this.historicalData.length - 1, this.currentSurfaceIndex + 1);
        this.updateSurfaceVisualization();
        this.updateUI();
    }
    
    resetToStart() {
        this.stopPlayback();
        this.currentSurfaceIndex = this.historicalData.length - 1;
        this.updateSurfaceVisualization();
        this.updateUI();
    }
    
    goToEnd() {
        this.stopPlayback();
        this.currentSurfaceIndex = 0;
        this.updateSurfaceVisualization();
        this.updateUI();
    }
    
    updatePlaybackButton() {
        const btn = document.getElementById('play-pause-btn');
        if (this.isPlaying) {
            btn.textContent = '⏸';
            btn.className = 'btn btn-danger';
        } else {
            btn.textContent = '▶';
            btn.className = 'btn btn-success';
        }
    }
    
    updateUI() {
        if (!this.historicalData.length) return;
        
        const surface = this.historicalData[this.currentSurfaceIndex];
        
        // Update frame counter
        const frameCounter = document.getElementById('frame-counter');
        if (frameCounter) {
            frameCounter.textContent = `Frame ${this.historicalData.length - this.currentSurfaceIndex} of ${this.historicalData.length}`;
        }
        
        // Update timeline slider
        const slider = document.getElementById('timeline-slider');
        if (slider) {
            slider.value = this.historicalData.length - 1 - this.currentSurfaceIndex;
        }
        
        // Update surface select
        this.populateSurfaceSelect();
        const surfaceSelect = document.getElementById('surface-select');
        if (surfaceSelect) {
            surfaceSelect.value = this.currentSurfaceIndex;
        }
        
        // Update surface information
        this.updateSurfaceInfo(surface);
        
        // Update playback button states
        const stepForwardBtn = document.getElementById('step-forward-btn');
        const stepBackwardBtn = document.getElementById('step-backward-btn');
        
        if (stepForwardBtn) {
            stepForwardBtn.disabled = this.currentSurfaceIndex === 0;
        }
        if (stepBackwardBtn) {
            stepBackwardBtn.disabled = this.currentSurfaceIndex === this.historicalData.length - 1;
        }
    }
    
    populateSurfaceSelect() {
        const select = document.getElementById('surface-select');
        select.innerHTML = '';
        
        this.historicalData.forEach((surface, index) => {
            const option = document.createElement('option');
            option.value = index;
            option.textContent = `${this.historicalData.length - index}: ${new Date(surface.timestamp).toLocaleString()}`;
            select.appendChild(option);
        });
    }
    
    updateSurfaceInfo(surface) {
        const assetName = this.selectedAsset ? 
            this.assets.find(a => a.id === this.selectedAsset)?.ticker || 'Unknown' : 
            'All Assets';
        
        // Helper function to safely update element text
        const updateElement = (id, text) => {
            const element = document.getElementById(id);
            if (element) {
                element.textContent = text;
            }
        };
        
        updateElement('frame-info', `${this.currentSurfaceIndex + 1} of ${this.historicalData.length}`);
        updateElement('timestamp-info', new Date(surface.timestamp).toLocaleString());
        updateElement('asset-info', assetName);
        updateElement('price-info', surface.spot_price ? 
            `$${surface.spot_price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : 'N/A');
        updateElement('status-info', this.isPlaying ? '▶ Playing' : '⏸ Paused');
        
        // Surface dimensions
        const uniqueMoneyness = [...new Set(surface.moneyness || [])];
        const uniqueDaysToExpiry = [...new Set(surface.daysToExpiry || [])];
        const validVols = (surface.impliedVols || []).filter(v => typeof v === 'number' && !isNaN(v));
        
        updateElement('points-info', surface.moneyness?.length || 0);
        updateElement('strikes-info', uniqueMoneyness.length);
        updateElement('expiries-info', uniqueDaysToExpiry.length);
        
        if (uniqueMoneyness.length > 0) {
            updateElement('moneyness-range', 
                `${Math.min(...uniqueMoneyness).toFixed(3)} - ${Math.max(...uniqueMoneyness).toFixed(3)}`);
        }
        
        if (uniqueDaysToExpiry.length > 0) {
            updateElement('dte-range', 
                `${Math.min(...uniqueDaysToExpiry).toFixed(1)} - ${Math.max(...uniqueDaysToExpiry).toFixed(1)} days`);
        }
        
        if (validVols.length > 0) {
            updateElement('vol-range', 
                `${(Math.min(...validVols) / 100).toFixed(1)}% - ${(Math.max(...validVols) / 100).toFixed(1)}%`);
        }
    }
    
    showPlaybackControls() {
        if (this.historicalData.length > 1) {
            document.getElementById('playback-section').style.display = 'block';
            document.getElementById('manual-section').style.display = 'block';
            document.getElementById('info-section').style.display = 'block';
        }
    }
    
    showLoading(show) {
        const loading = document.getElementById('loading');
        if (show) {
            loading.classList.add('show');
        } else {
            loading.classList.remove('show');
        }
    }
    
    showError(message) {
        console.error(message);
        // You could add a toast notification here
    }
    
    updateConnectionStatus(status) {
        const statusElement = document.getElementById('connection-status');
        if (statusElement) {
            statusElement.textContent = status;
            statusElement.className = `connection-status ${status.toLowerCase()}`;
        }
    }
    
    // Viridis color map function (similar to Plotly's Viridis)
    viridisColor(t) {
        // Clamp t to [0, 1]
        t = Math.max(0, Math.min(1, t));
        
        // Viridis color map coefficients
        const c0 = [0.267004, 0.004874, 0.329415];
        const c1 = [0.229739, 0.706468, 0.534389];
        const c2 = [0.127568, 0.566949, 0.550556];
        const c3 = [0.369214, 0.788888, 0.382914];
        const c4 = [0.993248, 0.906157, 0.143936];
        
        // Interpolate between color stops
        if (t < 0.25) {
            const u = t / 0.25;
            return {
                r: c0[0] * (1 - u) + c1[0] * u,
                g: c0[1] * (1 - u) + c1[1] * u,
                b: c0[2] * (1 - u) + c1[2] * u
            };
        } else if (t < 0.5) {
            const u = (t - 0.25) / 0.25;
            return {
                r: c1[0] * (1 - u) + c2[0] * u,
                g: c1[1] * (1 - u) + c2[1] * u,
                b: c1[2] * (1 - u) + c2[2] * u
            };
        } else if (t < 0.75) {
            const u = (t - 0.5) / 0.25;
            return {
                r: c2[0] * (1 - u) + c3[0] * u,
                g: c2[1] * (1 - u) + c3[1] * u,
                b: c2[2] * (1 - u) + c3[2] * u
            };
        } else {
            const u = (t - 0.75) / 0.25;
            return {
                r: c3[0] * (1 - u) + c4[0] * u,
                g: c3[1] * (1 - u) + c4[1] * u,
                b: c3[2] * (1 - u) + c4[2] * u
            };
        }
    }
    
    animate() {
        if (!this.isAnimating) {
            this.animationId = requestAnimationFrame(() => this.animate());
            return;
        }
        
        // No rotation - surface stays static
        
        // Render
        this.renderer.render(this.scene, this.camera);
        
        this.animationId = requestAnimationFrame(() => this.animate());
    }
}

// Initialize the viewer when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new HistoricalSurfaceViewer();
}); 