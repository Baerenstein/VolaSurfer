// VolaSurfer WebGL Wallpaper
class VolaSurferWallpaper {
    constructor() {
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.surfaceMesh = null;
        this.trendline = null;
        this.animationId = null;
        this.isAnimating = true;
        this.lastUpdateTime = 0;
        this.updateInterval = 5000; // 5 seconds
        
        // Interactive features
        this.raycaster = new THREE.Raycaster();
        this.pointer = new THREE.Vector2();
        this.intersection = null;
        this.spheres = [];
        this.spheresIndex = 0;
        
        // API endpoints
        this.apiBase = 'http://localhost:8000';
        this.endpoints = {
            surface: '/api/surface_snapshot',
            history: '/api/trendline',
            stats: '/api/stats',
            websocket: '/api/stream'
        };
        
        // WebSocket connection
        this.websocket = null;
        this.websocketReconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        
        this.init();
    }
    
    async init() {
        try {
            this.setupThreeJS();
            this.setupEventListeners();
            this.startTimeUpdate();
            await this.loadInitialData();
            this.hideLoading();
            this.animate();
        } catch (error) {
            console.error('Initialization error:', error);
            this.showError('Failed to initialize VolaSurfer Wallpaper');
        }
    }
    
    setupThreeJS() {
        const canvas = document.getElementById('webgl-canvas');
        
        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x000011);
        
        // Camera
        const aspect = window.innerWidth / window.innerHeight;
        this.camera = new THREE.PerspectiveCamera(75, aspect, 0.1, 1000);
        this.camera.position.set(5, 5, 5);
        this.camera.lookAt(0, 0, 0);
        
        // Renderer
        this.renderer = new THREE.WebGLRenderer({ 
            canvas: canvas,
            antialias: true,
            alpha: true
        });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        
        // Lighting
        this.setupLighting();
        
        // Grid helper
        const gridHelper = new THREE.GridHelper(10, 10, 0x444444, 0x222222);
        this.scene.add(gridHelper);
        
        // Handle window resize
        window.addEventListener('resize', () => this.onWindowResize());
        
        // Handle page unload
        window.addEventListener('beforeunload', () => {
            if (this.websocket) {
                this.websocket.close();
            }
        });
    }
    
    setupLighting() {
        // Ambient light - increased for better visibility
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        this.scene.add(ambientLight);
        
        // Directional light - main light source
        const directionalLight = new THREE.DirectionalLight(0xffffff, 1.0);
        directionalLight.position.set(10, 10, 5);
        directionalLight.castShadow = true;
        directionalLight.shadow.mapSize.width = 2048;
        directionalLight.shadow.mapSize.height = 2048;
        this.scene.add(directionalLight);
        
        // Additional directional light from opposite side
        const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.5);
        directionalLight2.position.set(-10, -10, -5);
        this.scene.add(directionalLight2);
        
        // Point light for dramatic effect
        const pointLight = new THREE.PointLight(0x00ff88, 0.8, 20);
        pointLight.position.set(0, 5, 0);
        this.scene.add(pointLight);
        
        // Rim light for better edge definition
        const rimLight = new THREE.DirectionalLight(0xffffff, 0.3);
        rimLight.position.set(0, 0, 10);
        this.scene.add(rimLight);
    }
    
    setupEventListeners() {
        // Control buttons
        document.getElementById('toggle-animation').addEventListener('click', () => {
            this.toggleAnimation();
        });
        
        document.getElementById('reset-camera').addEventListener('click', () => {
            this.resetCamera();
        });
        
        // Add WebSocket reconnect button
        const reconnectBtn = document.createElement('button');
        reconnectBtn.textContent = 'Reconnect WS';
        reconnectBtn.className = 'control-btn';
        reconnectBtn.addEventListener('click', () => {
            this.connectWebSocket();
        });
        
        // Add to controls
        const controls = document.querySelector('.controls');
        if (controls) {
            controls.appendChild(reconnectBtn);
        }
        
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
        
        // Pointer move for raycasting
        document.addEventListener('pointermove', (event) => {
            this.pointer.x = (event.clientX / window.innerWidth) * 2 - 1;
            this.pointer.y = -(event.clientY / window.innerHeight) * 2 + 1;
        });
        
        // Click for point selection
        document.addEventListener('click', (event) => {
            this.handlePointSelection();
        });
    }
    
    async loadInitialData() {
        try {
            // Load surface data
            const surfaceData = await this.fetchSurfaceData();
            if (surfaceData) {
                this.createSurfaceMesh(surfaceData);
            }
            
            // Load historical data for trendline
            const historyData = await this.fetchHistoryData();
            if (historyData && historyData.trendline) {
                this.createTrendline(historyData);
            }
            
            // Load initial stats
            const statsData = await this.fetchStats();
            if (statsData) {
                this.updateStats(statsData);
            }
            
            // Start WebSocket connection for real-time updates
            this.connectWebSocket();
            
            // Start periodic updates as fallback
            this.startPeriodicUpdates();
            
        } catch (error) {
            console.error('Error loading initial data:', error);
            this.showError('Failed to load market data');
        }
    }
    
    async fetchSurfaceData() {
        try {
            const response = await fetch(`${this.apiBase}${this.endpoints.surface}`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return await response.json();
        } catch (error) {
            console.error('Error fetching surface data:', error);
            return null;
        }
    }
    
    async fetchHistoryData() {
        try {
            const response = await fetch(`${this.apiBase}${this.endpoints.history}`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return await response.json();
        } catch (error) {
            console.error('Error fetching history data:', error);
            return null;
        }
    }
    
    async fetchStats() {
        try {
            const response = await fetch(`${this.apiBase}${this.endpoints.stats}`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return await response.json();
        } catch (error) {
            console.error('Error fetching stats:', error);
            return null;
        }
    }
    
    connectWebSocket() {
        try {
            // Close existing connection if any
            if (this.websocket) {
                this.websocket.close();
            }
            
            // Create WebSocket connection
            const wsUrl = `ws://localhost:8000${this.endpoints.websocket}`;
            this.websocket = new WebSocket(wsUrl);
            
            // Connection opened
            this.websocket.onopen = (event) => {
                console.log('WebSocket connected for real-time updates');
                this.websocketReconnectAttempts = 0;
                this.updateConnectionStatus('Connected');
            };
            
            // Listen for messages
            this.websocket.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleWebSocketMessage(data);
                } catch (error) {
                    console.error('Error parsing WebSocket message:', error);
                }
            };
            
            // Connection closed
            this.websocket.onclose = (event) => {
                console.log('WebSocket disconnected');
                this.updateConnectionStatus('Disconnected');
                
                // Attempt to reconnect if not manually closed
                if (!event.wasClean && this.websocketReconnectAttempts < this.maxReconnectAttempts) {
                    this.websocketReconnectAttempts++;
                    console.log(`Attempting to reconnect (${this.websocketReconnectAttempts}/${this.maxReconnectAttempts})...`);
                    setTimeout(() => this.connectWebSocket(), 3000);
                }
            };
            
            // Connection error
            this.websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateConnectionStatus('Error');
            };
            
        } catch (error) {
            console.error('Error connecting to WebSocket:', error);
            this.updateConnectionStatus('Error');
        }
    }
    
    handleWebSocketMessage(data) {
        const { type, data: messageData } = data;
        
        switch (type) {
            case 'surface_snapshot':
                if (messageData) {
                    this.createSurfaceMesh(messageData);
                }
                break;
                
            case 'stats':
                if (messageData) {
                    this.updateStats(messageData);
                }
                break;
                
            default:
                console.log('Unknown WebSocket message type:', type);
        }
    }
    
    updateConnectionStatus(status) {
        // Add a small indicator in the overlay
        const statusElement = document.getElementById('connection-status');
        if (!statusElement) {
            // Create status element if it doesn't exist
            const overlay = document.querySelector('.overlay-content');
            if (overlay) {
                const statusDiv = document.createElement('div');
                statusDiv.id = 'connection-status';
                statusDiv.style.cssText = `
                    position: absolute;
                    top: 10px;
                    left: 10px;
                    font-size: 10px;
                    padding: 2px 6px;
                    border-radius: 3px;
                    color: white;
                    font-weight: bold;
                `;
                overlay.appendChild(statusDiv);
            }
        }
        
        const element = document.getElementById('connection-status');
        if (element) {
            element.textContent = `WS: ${status}`;
            
            // Color coding
            switch (status) {
                case 'Connected':
                    element.style.background = '#00ff88';
                    break;
                case 'Disconnected':
                    element.style.background = '#ff8800';
                    break;
                case 'Error':
                    element.style.background = '#ff4444';
                    break;
                default:
                    element.style.background = '#666666';
            }
        }
    }
    
    handlePointSelection() {
        // Set up raycaster
        this.raycaster.setFromCamera(this.pointer, this.camera);
        this.raycaster.params.Points.threshold = 0.1;
        
        // Check intersections with point clouds
        const objects = [];
        if (this.surfaceMesh) objects.push(this.surfaceMesh);
        if (this.trendline) objects.push(this.trendline);
        
        const intersections = this.raycaster.intersectObjects(objects, false);
        this.intersection = intersections.length > 0 ? intersections[0] : null;
        
        if (this.intersection) {
            this.createSelectionSphere(this.intersection.point);
        }
    }
    
    createSelectionSphere(position) {
        // Create sphere geometry and material
        const sphereGeometry = new THREE.SphereGeometry(0.1, 16, 16);
        const sphereMaterial = new THREE.MeshBasicMaterial({ 
            color: 0xffff00,
            transparent: true,
            opacity: 0.8
        });
        
        // Create sphere mesh
        const sphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
        sphere.position.copy(position);
        this.scene.add(sphere);
        
        // Add to spheres array
        this.spheres.push(sphere);
        
        // Limit number of spheres
        if (this.spheres.length > 20) {
            const oldSphere = this.spheres.shift();
            this.scene.remove(oldSphere);
        }
    }
    
    createSurfaceMesh(surfaceData) {
        // Remove existing surface mesh
        if (this.surfaceMesh) {
            this.scene.remove(this.surfaceMesh);
        }
        
        const { x, y, z } = surfaceData;
        
        if (!x || !y || !z || x.length === 0) {
            console.warn('Invalid surface data structure');
            return;
        }
        
        // Find min/max values for scaling
        const minX = Math.min(...x);
        const maxX = Math.max(...x);
        const minY = Math.min(...y);
        const maxY = Math.max(...y);
        const minZ = Math.min(...z);
        const maxZ = Math.max(...z);
        
        // Create BufferGeometry for point cloud
        const geometry = new THREE.BufferGeometry();
        const numPoints = x.length;
        
        // Create position and color arrays
        const positions = new Float32Array(numPoints * 3);
        const colors = new Float32Array(numPoints * 3);
        
        // Scale factors for visualization
        const scaleX = 8 / (maxX - minX);
        const scaleY = 8 / (maxY - minY);
        const scaleZ = 4; // Scale volatility for visual effect
        
        // Fill the arrays with data
        for (let i = 0; i < numPoints; i++) {
            const index = i * 3;
            
            // Position: scale and center the data
            const scaledX = (x[i] - minX) * scaleX - 4;
            const scaledY = (y[i] - minY) * scaleY - 4;
            const scaledZ = (z[i] - minZ) * scaleZ;
            
            positions[index] = scaledX;
            positions[index + 1] = scaledY;
            positions[index + 2] = scaledZ;
            
            // Color: based on volatility (green to red gradient)
            const normalizedZ = (z[i] - minZ) / (maxZ - minZ);
            const intensity = normalizedZ * 0.8 + 0.2; // Ensure minimum brightness
            
            // Create a gradient from green (low volatility) to red (high volatility)
            colors[index] = normalizedZ * intensity; // Red component
            colors[index + 1] = (1 - normalizedZ) * intensity; // Green component
            colors[index + 2] = 0.2 * intensity; // Blue component (subtle)
        }
        
        // Set geometry attributes
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        geometry.computeBoundingBox();
        
        // Create PointsMaterial for the point cloud
        const material = new THREE.PointsMaterial({
            size: 0.1, // Point size
            vertexColors: true, // Use vertex colors
            transparent: true,
            opacity: 0.8,
            sizeAttenuation: true, // Points get smaller with distance
            blending: THREE.AdditiveBlending // Additive blending for glow effect
        });
        
        // Create the point cloud
        this.surfaceMesh = new THREE.Points(geometry, material);
        this.surfaceMesh.rotation.x = -Math.PI / 2; // Rotate to horizontal
        this.surfaceMesh.castShadow = false; // Points don't cast shadows
        this.surfaceMesh.receiveShadow = false;
        
        this.scene.add(this.surfaceMesh);
        
        // Store the data for potential interactions
        this.surfaceMesh.userData = {
            originalData: { x, y, z },
            bounds: { minX, maxX, minY, maxY, minZ, maxZ }
        };
    }
    
    createTrendline(historyData) {
        // Remove existing trendline
        if (this.trendline) {
            this.scene.remove(this.trendline);
        }
        
        const { trendline } = historyData;
        
        if (!trendline || trendline.length === 0) {
            console.warn('No trendline data available');
            return;
        }
        
        // Create BufferGeometry for trendline point cloud
        const geometry = new THREE.BufferGeometry();
        const numPoints = trendline.length;
        
        // Create position and color arrays
        const positions = new Float32Array(numPoints * 3);
        const colors = new Float32Array(numPoints * 3);
        
        // Find min/max for scaling
        const prices = trendline.map(p => p.price);
        const minPrice = Math.min(...prices);
        const maxPrice = Math.max(...prices);
        
        // Fill the arrays with trendline data
        for (let i = 0; i < numPoints; i++) {
            const index = i * 3;
            const point = trendline[i];
            
            // Position: time on X, price on Y, fixed Z
            const x = (i / numPoints) * 8 - 4; // X: time
            const y = ((point.price - minPrice) / (maxPrice - minPrice)) * 4; // Y: normalized price
            const z = 1.0; // Z: above the surface
            
            positions[index] = x;
            positions[index + 1] = y;
            positions[index + 2] = z;
            
            // Color: gradient from orange to yellow based on time
            const timeProgress = i / numPoints;
            colors[index] = 1.0; // Red component (full)
            colors[index + 1] = 0.5 + timeProgress * 0.5; // Green component (increasing)
            colors[index + 2] = 0.0; // Blue component (none)
        }
        
        // Set geometry attributes
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        geometry.computeBoundingBox();
        
        // Create PointsMaterial for the trendline
        const material = new THREE.PointsMaterial({
            size: 0.15, // Larger points for trendline
            vertexColors: true,
            transparent: true,
            opacity: 0.9,
            sizeAttenuation: true,
            blending: THREE.AdditiveBlending
        });
        
        // Create the trendline point cloud
        this.trendline = new THREE.Points(geometry, material);
        this.scene.add(this.trendline);
        
        // Store the data
        this.trendline.userData = {
            originalData: trendline,
            bounds: { minPrice, maxPrice }
        };
    }
    
    startPeriodicUpdates() {
        setInterval(async () => {
            // Only update via HTTP if WebSocket is not connected
            if (this.isAnimating && (!this.websocket || this.websocket.readyState !== WebSocket.OPEN)) {
                await this.updateData();
            }
        }, this.updateInterval);
    }
    
    async updateData() {
        try {
            const surfaceData = await this.fetchSurfaceData();
            if (surfaceData) {
                this.createSurfaceMesh(surfaceData);
            }
            
            const statsData = await this.fetchStats();
            if (statsData) {
                this.updateStats(statsData);
            }
        } catch (error) {
            console.error('Error updating data:', error);
        }
    }
    
    updateStats(statsData) {
        // Update spread from API
        if (statsData.spread !== undefined) {
            document.getElementById('spread-value').textContent = `${statsData.spread}%`;
        }
        
        // Update volume from API
        if (statsData.volume !== undefined) {
            document.getElementById('volume-value').textContent = statsData.volume.toString();
        }
        
        // Update last update time
        const now = new Date();
        document.getElementById('last-update').textContent = now.toLocaleTimeString();
    }
    
    startTimeUpdate() {
        setInterval(() => {
            const now = new Date();
            document.getElementById('current-time').textContent = now.toLocaleTimeString();
        }, 1000);
    }
    
    animate() {
        if (!this.isAnimating) {
            this.animationId = requestAnimationFrame(() => this.animate());
            return;
        }
        
        // Rotate surface mesh slowly
        if (this.surfaceMesh) {
            this.surfaceMesh.rotation.y += 0.005;
        }
        
        // Animate trendline if it exists
        if (this.trendline) {
            this.trendline.rotation.y += 0.002;
        }
        
        // Animate selection spheres
        this.spheres.forEach(sphere => {
            sphere.scale.multiplyScalar(0.98);
            sphere.scale.clampScalar(0.01, 1);
        });
        
        // Render
        this.renderer.render(this.scene, this.camera);
        
        this.animationId = requestAnimationFrame(() => this.animate());
    }
    
    toggleAnimation() {
        this.isAnimating = !this.isAnimating;
        const button = document.getElementById('toggle-animation');
        button.textContent = this.isAnimating ? 'Pause' : 'Play';
        
        if (this.isAnimating && !this.animationId) {
            this.animate();
        }
    }
    
    resetCamera() {
        this.camera.position.set(5, 5, 5);
        this.camera.lookAt(0, 0, 0);
    }
    
    onWindowResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
    }
    
    hideLoading() {
        const loading = document.getElementById('loading');
        loading.classList.add('hidden');
        setTimeout(() => {
            loading.style.display = 'none';
        }, 500);
    }
    
    showError(message) {
        const loading = document.getElementById('loading');
        const loadingContent = loading.querySelector('.loading-content');
        loadingContent.innerHTML = `
            <div style="color: #ff4444; font-size: 18px; margin-bottom: 10px;">⚠️</div>
            <p style="color: #ff4444;">${message}</p>
            <button onclick="location.reload()" style="margin-top: 15px; padding: 8px 16px; background: #ff4444; border: none; color: white; border-radius: 4px; cursor: pointer;">Retry</button>
        `;
    }
}

// Initialize the wallpaper when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new VolaSurferWallpaper();
}); 