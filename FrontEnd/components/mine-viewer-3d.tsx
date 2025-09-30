"use client"

import { useEffect, useRef, useState } from "react"
import * as THREE from "three"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { RotateCcw, ZoomIn, ZoomOut, Camera } from "lucide-react"
import { loadMineModel, loadBinghamCanyonMine } from "@/utils/loadMineModel" // Import the loadMineModel function

interface Sensor {
  id: string
  name: string
  type: "radar" | "piezometer" | "extensometer" | "seismometer" | "weather"
  position: [number, number, number]
  location: string
  status: "online" | "offline" | "warning" | "critical"
  readings: {
    displacement?: number
    velocity?: number
    pressure?: number
    strain?: number
    vibration?: number
    rainfall?: number
    temperature?: number
  }
  lastUpdate: string
}

interface MineViewer3DProps {
  sensors?: Sensor[]
  showHeatmap?: boolean
  heatmapType?: "slope" | "rockfall"
  onSensorHover?: (sensor: Sensor | null) => void
}

export function MineViewer3D({
  sensors = [],
  showHeatmap = false,
  heatmapType = "slope",
  onSensorHover,
}: MineViewer3DProps) {
  const mountRef = useRef<HTMLDivElement>(null)
  const sceneRef = useRef<THREE.Scene>()
  const rendererRef = useRef<THREE.WebGLRenderer>()
  const cameraRef = useRef<THREE.PerspectiveCamera>()
  const frameRef = useRef<number>()
  const controlsRef = useRef<any>()
  const [hoveredSensor, setHoveredSensor] = useState<Sensor | null>(null)
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 })
  const [isAutoRotating, setIsAutoRotating] = useState(true)
  const [isModelLoading, setIsModelLoading] = useState(false)
  const [modelError, setModelError] = useState<string | null>(null)

  useEffect(() => {
    if (!mountRef.current) return

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x87ceeb)
    scene.fog = new THREE.Fog(0x87ceeb, 50, 200)
    sceneRef.current = scene

    const camera = new THREE.PerspectiveCamera(
      75,
      mountRef.current.clientWidth / mountRef.current.clientHeight,
      0.1,
      1000,
    )
    // Adjusted camera position for Bingham Canyon model - zoomed out for full view
    camera.position.set(100, 80, 100)
    camera.lookAt(0, 0, 0)
    cameraRef.current = camera

    const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true })
    renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight)
    renderer.shadowMap.enabled = true
    renderer.shadowMap.type = THREE.PCFSoftShadowMap
    renderer.setClearColor(0x87ceeb)
    rendererRef.current = renderer
    mountRef.current.appendChild(renderer.domElement)

    let isDragging = false
    let previousMousePosition = { x: 0, y: 0 }
    let cameraDistance = 150 // Zoomed out for better overview
    let cameraAngleX = 0
    let cameraAngleY = 0.8

    const updateCameraPosition = () => {
      camera.position.x = Math.cos(cameraAngleX) * Math.cos(cameraAngleY) * cameraDistance
      camera.position.y = Math.sin(cameraAngleY) * cameraDistance
      camera.position.z = Math.sin(cameraAngleX) * Math.cos(cameraAngleY) * cameraDistance
      camera.lookAt(0, 0, 0)
    }

    const onMouseDown = (event: MouseEvent) => {
      isDragging = true
      previousMousePosition = { x: event.clientX, y: event.clientY }
      setIsAutoRotating(false)
    }

    const onMouseMove = (event: MouseEvent) => {
      const rect = renderer.domElement.getBoundingClientRect()
      const mouse = new THREE.Vector2()
      mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1
      mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1

      setMousePosition({ x: event.clientX, y: event.clientY })

      if (isDragging) {
        const deltaX = event.clientX - previousMousePosition.x
        const deltaY = event.clientY - previousMousePosition.y

        cameraAngleX += deltaX * 0.01
        cameraAngleY = Math.max(-Math.PI / 2 + 0.1, Math.min(Math.PI / 2 - 0.1, cameraAngleY + deltaY * 0.01))

        updateCameraPosition()
        previousMousePosition = { x: event.clientX, y: event.clientY }
      } else {
        const raycaster = new THREE.Raycaster()
        raycaster.setFromCamera(mouse, camera)
        const intersects = raycaster.intersectObjects(sensorMeshes)

        if (intersects.length > 0) {
          const intersectedObject = intersects[0].object
          const sensor = sensors.find((s) => s.id === intersectedObject.userData.sensorId)
          if (sensor) {
            setHoveredSensor(sensor)
            onSensorHover?.(sensor)
            renderer.domElement.style.cursor = "pointer"
          }
        } else {
          setHoveredSensor(null)
          onSensorHover?.(null)
          renderer.domElement.style.cursor = isDragging ? "grabbing" : "grab"
        }
      }
    }

    const onMouseUp = () => {
      isDragging = false
      renderer.domElement.style.cursor = "grab"
    }

    const onWheel = (event: WheelEvent) => {
      event.preventDefault()
      cameraDistance = Math.max(50, Math.min(300, cameraDistance + event.deltaY * 0.01))
      updateCameraPosition()
    }

    renderer.domElement.addEventListener("mousedown", onMouseDown)
    renderer.domElement.addEventListener("mousemove", onMouseMove)
    renderer.domElement.addEventListener("mouseup", onMouseUp)
    renderer.domElement.addEventListener("wheel", onWheel)
    renderer.domElement.style.cursor = "grab"

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.4)
    scene.add(ambientLight)

    const directionalLight = new THREE.DirectionalLight(0xffffff, 1.2)
    directionalLight.position.set(50, 50, 25)
    directionalLight.castShadow = true
    directionalLight.shadow.mapSize.width = 4096
    directionalLight.shadow.mapSize.height = 4096
    directionalLight.shadow.camera.near = 0.5
    directionalLight.shadow.camera.far = 200
    directionalLight.shadow.camera.left = -50
    directionalLight.shadow.camera.right = 50
    directionalLight.shadow.camera.top = 50
    directionalLight.shadow.camera.bottom = -50
    scene.add(directionalLight)

    const secondaryLight = new THREE.DirectionalLight(0xffffff, 0.6)
    secondaryLight.position.set(-30, 30, -30)
    scene.add(secondaryLight)

    // Load the Bingham Canyon Mine model by default
    loadBinghamCanyonMine(scene, setIsModelLoading, setModelError)

    // Add some custom sensors with guaranteed positive Y coordinates
    createCustomSensors(scene)

    const sensorMeshes = createAdvancedSensorMarkers(scene, sensors, true) // Pass true for Bingham Canyon scaling

    if (showHeatmap) {
      createAdvancedHeatmapOverlay(scene, heatmapType, true) // Pass true for Bingham Canyon scaling
    }

    const animate = () => {
      frameRef.current = requestAnimationFrame(animate)

      if (isAutoRotating) {
        cameraAngleX += 0.005
        updateCameraPosition()
      }

      sensorMeshes.forEach((mesh, index) => {
        const sensor = sensors[index]
        if (sensor?.status === "online") {
          const time = Date.now() * 0.003 + index
          mesh.scale.setScalar(1 + Math.sin(time) * 0.15)
        }
      })

      renderer.render(scene, camera)
    }
    animate()

    const handleResize = () => {
      if (!mountRef.current) return

      camera.aspect = mountRef.current.clientWidth / mountRef.current.clientHeight
      camera.updateProjectionMatrix()
      renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight)
    }

    window.addEventListener("resize", handleResize)

    controlsRef.current = {
      resetView: () => {
        cameraAngleX = 0
        cameraAngleY = 0.8
        cameraDistance = 150
        updateCameraPosition()
        setIsAutoRotating(true)
      },
      zoomIn: () => {
        cameraDistance = Math.max(50, cameraDistance - 10)
        updateCameraPosition()
      },
      zoomOut: () => {
        cameraDistance = Math.min(300, cameraDistance + 10)
        updateCameraPosition()
      },
      screenshot: () => {
        renderer.render(scene, camera)
        const dataURL = renderer.domElement.toDataURL("image/png")
        const link = document.createElement("a")
        link.download = `mine-view-${Date.now()}.png`
        link.href = dataURL
        link.click()
      },
      loadCustomModel: (modelPath: string) => {
        setIsModelLoading(true)
        setModelError(null)
        loadMineModel(scene, modelPath, setIsModelLoading, setModelError)
      },
    }

    return () => {
      if (frameRef.current) {
        cancelAnimationFrame(frameRef.current)
      }
      renderer.domElement.removeEventListener("mousedown", onMouseDown)
      renderer.domElement.removeEventListener("mousemove", onMouseMove)
      renderer.domElement.removeEventListener("mouseup", onMouseUp)
      renderer.domElement.removeEventListener("wheel", onWheel)
      window.removeEventListener("resize", handleResize)
      if (mountRef.current && renderer.domElement) {
        mountRef.current.removeChild(renderer.domElement)
      }
      renderer.dispose()
    }
  }, [sensors, showHeatmap, heatmapType, onSensorHover, isAutoRotating])

  const handleResetView = () => {
    controlsRef.current?.resetView()
  }

  const handleZoomIn = () => {
    controlsRef.current?.zoomIn()
  }

  const handleZoomOut = () => {
    controlsRef.current?.zoomOut()
  }

  const handleScreenshot = () => {
    controlsRef.current?.screenshot()
  }

  const handleLoadModel = () => {
    const input = document.createElement("input")
    input.type = "file"
    input.accept = ".glb,.gltf"
    input.onchange = (e) => {
      const file = (e.target as HTMLInputElement).files?.[0]
      if (file) {
        const url = URL.createObjectURL(file)
        controlsRef.current?.loadCustomModel(url)
      }
    }
    input.click()
  }

  return (
    <div className="relative w-full h-full">
      <div ref={mountRef} className="w-full h-full rounded-lg overflow-hidden" />

      {isModelLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-background/80 backdrop-blur-sm">
          <div className="text-center space-y-2">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto"></div>
            <p className="text-sm text-muted-foreground">Loading 3D model...</p>
          </div>
        </div>
      )}

      {modelError && (
        <div className="absolute inset-0 flex items-center justify-center bg-background/80 backdrop-blur-sm">
          <div className="text-center space-y-2 max-w-md p-4">
            <p className="text-sm text-destructive">Failed to load 3D model</p>
            <p className="text-xs text-muted-foreground">{modelError}</p>
            <Button size="sm" onClick={() => setModelError(null)}>
              Dismiss
            </Button>
          </div>
        </div>
      )}

      <div className="absolute top-2 right-2 flex gap-1">
        <Button size="sm" variant="secondary" onClick={handleLoadModel} title="Load Custom Model">
          📁
        </Button>
        <Button size="sm" variant="secondary" onClick={handleResetView} title="Reset View">
          <RotateCcw className="h-3 w-3" />
        </Button>
        <Button size="sm" variant="secondary" onClick={handleZoomIn} title="Zoom In">
          <ZoomIn className="h-3 w-3" />
        </Button>
        <Button size="sm" variant="secondary" onClick={handleZoomOut} title="Zoom Out">
          <ZoomOut className="h-3 w-3" />
        </Button>
        <Button size="sm" variant="secondary" onClick={handleScreenshot} title="Screenshot">
          <Camera className="h-3 w-3" />
        </Button>
      </div>

      <div className="absolute bottom-2 left-2">
        <Badge variant={isAutoRotating ? "default" : "secondary"} className="text-xs">
          {isAutoRotating ? "Auto-rotating" : "Manual control"}
        </Badge>
      </div>

      {hoveredSensor && (
        <div
          className="absolute z-10 bg-card border border-border rounded-lg p-3 shadow-lg pointer-events-none max-w-xs"
          style={{
            left: Math.min(mousePosition.x + 10, window.innerWidth - 250),
            top: mousePosition.y - 10,
            transform: "translate(0, -100%)",
          }}
        >
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <Badge
                variant={
                  hoveredSensor.status === "online"
                    ? "default"
                    : hoveredSensor.status === "warning"
                      ? "secondary"
                      : hoveredSensor.status === "critical"
                        ? "destructive"
                        : "outline"
                }
              >
                {hoveredSensor.status}
              </Badge>
              <span className="font-medium text-sm">{hoveredSensor.id}</span>
              <SensorTypeIcon type={hoveredSensor.type} />
            </div>
            <p className="text-xs text-muted-foreground font-medium">{hoveredSensor.type.toUpperCase()}</p>
            <p className="text-xs text-muted-foreground">{hoveredSensor.location}</p>
            {hoveredSensor.status !== "offline" && (
              <div className="grid grid-cols-1 gap-1 text-xs">
                {hoveredSensor.readings.displacement !== undefined && (
                  <div className="flex justify-between">
                    <span>Displacement:</span>
                    <span className="font-medium">{hoveredSensor.readings.displacement}mm</span>
                  </div>
                )}
                {hoveredSensor.readings.velocity !== undefined && (
                  <div className="flex justify-between">
                    <span>Velocity:</span>
                    <span className="font-medium">{hoveredSensor.readings.velocity}mm/day</span>
                  </div>
                )}
                {hoveredSensor.readings.pressure !== undefined && (
                  <div className="flex justify-between">
                    <span>Pressure:</span>
                    <span className="font-medium">{hoveredSensor.readings.pressure}kPa</span>
                  </div>
                )}
                {hoveredSensor.readings.strain !== undefined && (
                  <div className="flex justify-between">
                    <span>Strain:</span>
                    <span className="font-medium">{hoveredSensor.readings.strain}%</span>
                  </div>
                )}
                {hoveredSensor.readings.vibration !== undefined && (
                  <div className="flex justify-between">
                    <span>Vibration:</span>
                    <span className="font-medium">{hoveredSensor.readings.vibration}mm/s²</span>
                  </div>
                )}
                {hoveredSensor.readings.rainfall !== undefined && (
                  <div className="flex justify-between">
                    <span>Rainfall:</span>
                    <span className="font-medium">{hoveredSensor.readings.rainfall}mm</span>
                  </div>
                )}
                {hoveredSensor.readings.temperature !== undefined && (
                  <div className="flex justify-between">
                    <span>Temperature:</span>
                    <span className="font-medium">{hoveredSensor.readings.temperature}°C</span>
                  </div>
                )}
              </div>
            )}
            <p className="text-xs text-muted-foreground">Updated: {hoveredSensor.lastUpdate}</p>
          </div>
        </div>
      )}
    </div>
  )
}

function SensorTypeIcon({ type }: { type: string }) {
  const icons = {
    radar: "📡",
    piezometer: "💧",
    extensometer: "📏",
    seismometer: "📊",
    weather: "🌤️",
  }
  return <span className="text-xs">{icons[type as keyof typeof icons] || "📍"}</span>
}

function createRealisticMinePit(scene: THREE.Scene) {
  const mineGroup = new THREE.Group()
  mineGroup.name = "mine-structure"

  // Create textured materials
  const rockMaterial = new THREE.MeshLambertMaterial({
    color: 0x8b7355,
    roughness: 0.8,
  })

  const roadMaterial = new THREE.MeshLambertMaterial({
    color: 0x696969,
    roughness: 0.6,
  })

  const vegetationMaterial = new THREE.MeshLambertMaterial({
    color: 0x228b22,
    roughness: 0.9,
  })

  const waterMaterial = new THREE.MeshLambertMaterial({
    color: 0x4169e1,
    transparent: true,
    opacity: 0.7,
  })

  // Create terraced mine pit with multiple benches
  const pitRadius = 30
  const benchHeight = 3
  const benchWidth = 4
  const numBenches = 8

  for (let i = 0; i < numBenches; i++) {
    const currentRadius = pitRadius - i * benchWidth
    const benchY = -(i * benchHeight)

    // Create bench geometry
    const benchGeometry = new THREE.RingGeometry(Math.max(currentRadius - benchWidth, 2), currentRadius, 32)
    const bench = new THREE.Mesh(benchGeometry, rockMaterial)
    bench.rotation.x = -Math.PI / 2
    bench.position.y = benchY
    bench.receiveShadow = true
    mineGroup.add(bench)

    // Create bench walls
    const wallGeometry = new THREE.CylinderGeometry(currentRadius, currentRadius, benchHeight, 32, 1, true)
    const wall = new THREE.Mesh(wallGeometry, rockMaterial)
    wall.position.y = benchY - benchHeight / 2
    wall.castShadow = true
    wall.receiveShadow = true
    mineGroup.add(wall)

    // Add haul roads on alternating benches
    if (i % 2 === 0 && i < numBenches - 2) {
      const roadGeometry = new THREE.RingGeometry(currentRadius - 1, currentRadius, 32, 1, 0, Math.PI)
      const road = new THREE.Mesh(roadGeometry, roadMaterial)
      road.rotation.x = -Math.PI / 2
      road.position.y = benchY + 0.1
      mineGroup.add(road)
    }
  }

  // Add water at the bottom of the pit
  const waterGeometry = new THREE.CircleGeometry(6, 32)
  const water = new THREE.Mesh(waterGeometry, waterMaterial)
  water.rotation.x = -Math.PI / 2
  water.position.y = -(numBenches * benchHeight) + 0.5
  mineGroup.add(water)

  // Add surrounding terrain and vegetation
  const terrainGeometry = new THREE.RingGeometry(pitRadius + 5, pitRadius + 20, 32)
  const terrain = new THREE.Mesh(terrainGeometry, vegetationMaterial)
  terrain.rotation.x = -Math.PI / 2
  terrain.position.y = 2
  terrain.receiveShadow = true
  mineGroup.add(terrain)

  // Add mining equipment and structures
  createMiningEquipment(mineGroup, rockMaterial)

  // Add rock piles around the pit
  for (let i = 0; i < 8; i++) {
    const angle = (i / 8) * Math.PI * 2
    const distance = pitRadius + 8 + Math.random() * 5
    const pileGeometry = new THREE.ConeGeometry(2 + Math.random() * 2, 3 + Math.random() * 3, 8)
    const pile = new THREE.Mesh(pileGeometry, rockMaterial)
    pile.position.set(Math.cos(angle) * distance, 1.5 + Math.random(), Math.sin(angle) * distance)
    pile.castShadow = true
    mineGroup.add(pile)
  }

  scene.add(mineGroup)
}

function createMiningEquipment(mineGroup: THREE.Group, material: THREE.Material) {
  // Add excavator
  const excavatorBody = new THREE.BoxGeometry(3, 1.5, 2)
  const excavator = new THREE.Mesh(excavatorBody, material)
  excavator.position.set(-15, -6, 8)
  excavator.castShadow = true
  mineGroup.add(excavator)

  // Add dump trucks
  for (let i = 0; i < 3; i++) {
    const truckGeometry = new THREE.BoxGeometry(2, 1, 4)
    const truck = new THREE.Mesh(truckGeometry, material)
    const angle = (i / 3) * Math.PI * 2
    truck.position.set(Math.cos(angle) * 20, -3, Math.sin(angle) * 20)
    truck.castShadow = true
    mineGroup.add(truck)
  }

  // Add conveyor belt structure
  const conveyorGeometry = new THREE.BoxGeometry(1, 0.5, 25)
  const conveyor = new THREE.Mesh(conveyorGeometry, material)
  conveyor.position.set(25, 2, 0)
  conveyor.rotation.z = -0.2
  conveyor.castShadow = true
  mineGroup.add(conveyor)
}

function createAdvancedSensorMarkers(scene: THREE.Scene, sensors: Sensor[], isBinghamCanyon: boolean = false): THREE.Mesh[] {
  const sensorMeshes: THREE.Mesh[] = []

  sensors.forEach((sensor) => {
    let geometry: THREE.BufferGeometry
    let color: number

    // Different shapes for different sensor types
    switch (sensor.type) {
      case "radar":
        geometry = new THREE.ConeGeometry(0.3, 0.8, 8)
        break
      case "piezometer":
        geometry = new THREE.CylinderGeometry(0.2, 0.2, 0.8, 8)
        break
      case "extensometer":
        geometry = new THREE.BoxGeometry(0.3, 0.8, 0.3)
        break
      case "seismometer":
        geometry = new THREE.OctahedronGeometry(0.4)
        break
      case "weather":
        geometry = new THREE.SphereGeometry(0.4, 12, 12)
        break
      default:
        geometry = new THREE.SphereGeometry(0.3, 16, 16)
    }

    // Color based on status
    switch (sensor.status) {
      case "online":
        color = 0x22c55e // Green
        break
      case "warning":
        color = 0xeab308 // Yellow
        break
      case "critical":
        color = 0xef4444 // Red
        break
      default:
        color = 0x6b7280 // Gray
    }

    const material = new THREE.MeshLambertMaterial({
      color: color,
      emissive: color,
      emissiveIntensity: sensor.status === "online" ? 0.3 : sensor.status === "critical" ? 0.5 : 0.2,
    })

    const sensorMesh = new THREE.Mesh(geometry, material)
    
    // Scale sensor positions for Bingham Canyon model
    const scaleFactor = isBinghamCanyon ? 2 : 1
    const scaledPosition = sensor.position.map(coord => coord * scaleFactor) as [number, number, number]
    sensorMesh.position.set(...scaledPosition)
    
    // Scale sensor size for Bingham Canyon model
    if (isBinghamCanyon) {
      sensorMesh.scale.setScalar(2)
    }
    
    sensorMesh.userData = { sensorId: sensor.id }
    sensorMesh.castShadow = true

    // Add pulsing wireframe for active sensors
    if (sensor.status !== "offline") {
      const wireframeGeometry = geometry.clone()
      const wireframeMaterial = new THREE.MeshBasicMaterial({
        color: color,
        wireframe: true,
        transparent: true,
        opacity: 0.4,
      })
      const wireframe = new THREE.Mesh(wireframeGeometry, wireframeMaterial)
      wireframe.position.copy(sensorMesh.position)
      wireframe.scale.setScalar(isBinghamCanyon ? 2.4 : 1.2)
      scene.add(wireframe)
    }

    scene.add(sensorMesh)
    sensorMeshes.push(sensorMesh)
  })

  return sensorMeshes
}

function createAdvancedHeatmapOverlay(scene: THREE.Scene, type: "slope" | "rockfall", isBinghamCanyon: boolean = false) {
  const zones =
    type === "slope"
      ? [
          { position: [-8, 8, 5], color: 0x22c55e, opacity: 0.4, size: 4, risk: "low" },
          { position: [10, 8, -8], color: 0xeab308, opacity: 0.5, size: 5, risk: "medium" },
          { position: [-12, 8, -10], color: 0xf97316, opacity: 0.6, size: 4.5, risk: "high" },
          { position: [15, 8, 12], color: 0xef4444, opacity: 0.7, size: 6, risk: "critical" },
          { position: [0, 8, 0], color: 0xeab308, opacity: 0.4, size: 3, risk: "medium" },
        ] 
      : [
          { position: [-5, 8, 8], color: 0xef4444, opacity: 0.8, size: 4, risk: "critical" },
          { position: [12, 8, -5], color: 0xf97316, opacity: 0.6, size: 3.5, risk: "high" },
          { position: [-15, 8, -12], color: 0xeab308, opacity: 0.5, size: 4, risk: "medium" },
          { position: [8, 8, 15], color: 0x22c55e, opacity: 0.4, size: 3, risk: "low" },
          { position: [0, 8, 0], color: 0xf97316, opacity: 0.7, size: 5, risk: "high" },
        ]

  zones.forEach((zone, index) => {
    // Scale zone size and position for Bingham Canyon model
    const scaleFactor = isBinghamCanyon ? 2 : 1
    const scaledSize = zone.size * scaleFactor
    const scaledPosition = zone.position.map(coord => coord * scaleFactor) as [number, number, number]
    
    const geometry = new THREE.SphereGeometry(scaledSize, 16, 16)
    const material = new THREE.MeshLambertMaterial({
      color: zone.color,
      transparent: true,
      opacity: zone.opacity,
      depthWrite: false,
    })

    const heatmapZone = new THREE.Mesh(geometry, material)
    heatmapZone.position.set(...scaledPosition)

    // Animated pulsing effect
    const animate = () => {
      const time = Date.now() * 0.001 + index
      heatmapZone.scale.setScalar(1 + Math.sin(time) * 0.15)
      heatmapZone.material.opacity = zone.opacity + Math.sin(time * 2) * 0.1
      requestAnimationFrame(animate)
    }
    animate()

    scene.add(heatmapZone)
  })
}

function createCustomSensors(scene: THREE.Scene) {
  // Custom sensor positions - spread apart across the mine, raised above surface
  const customSensors = [
    { position: [-25, 12, 20], color: 0x22c55e, type: "radar" },
    { position: [30, 10, -15], color: 0xeab308, type: "piezometer" },
    { position: [-20, 14, -25], color: 0xef4444, type: "extensometer" },
    { position: [25, 9, 30], color: 0x3b82f6, type: "seismometer" },
    { position: [0, 16, 0], color: 0x8b5cf6, type: "weather" },
    { position: [-35, 11, 10], color: 0xf97316, type: "inclinometer" },
    { position: [35, 10, -10], color: 0x06b6d4, type: "crackmeter" },
    { position: [15, 13, 35], color: 0x10b981, type: "tiltmeter" },
  ]

  customSensors.forEach((sensor, index) => {
    let geometry: THREE.BufferGeometry

    // Different shapes for different sensor types - made much larger for visibility
    switch (sensor.type) {
      case "radar":
        geometry = new THREE.ConeGeometry(2, 6, 8)
        break
      case "piezometer":
        geometry = new THREE.CylinderGeometry(1.5, 1.5, 6, 8)
        break
      case "extensometer":
        geometry = new THREE.BoxGeometry(2, 6, 2)
        break
      case "seismometer":
        geometry = new THREE.OctahedronGeometry(3)
        break
      case "weather":
        geometry = new THREE.SphereGeometry(3, 12, 12)
        break
      case "inclinometer":
        geometry = new THREE.TetrahedronGeometry(2.5)
        break
      case "crackmeter":
        geometry = new THREE.DodecahedronGeometry(2)
        break
      case "tiltmeter":
        geometry = new THREE.IcosahedronGeometry(2.2)
        break
      default:
        geometry = new THREE.SphereGeometry(2, 16, 16)
    }

    const material = new THREE.MeshLambertMaterial({
      color: sensor.color,
      emissive: sensor.color,
      emissiveIntensity: 0.6, // Increased for better visibility
    })

    const sensorMesh = new THREE.Mesh(geometry, material)
    sensorMesh.position.set(...sensor.position)
    sensorMesh.castShadow = true

    // Add pulsing animation
    const animate = () => {
      const time = Date.now() * 0.003 + index
      sensorMesh.scale.setScalar(1 + Math.sin(time) * 0.15)
      requestAnimationFrame(animate)
    }
    animate()

    // Add wireframe for better visibility
    const wireframeGeometry = geometry.clone()
    const wireframeMaterial = new THREE.MeshBasicMaterial({
      color: sensor.color,
      wireframe: true,
      transparent: true,
      opacity: 0.4,
    })
    const wireframe = new THREE.Mesh(wireframeGeometry, wireframeMaterial)
    wireframe.position.copy(sensorMesh.position)
    wireframe.scale.setScalar(1.3)
    scene.add(wireframe)

    scene.add(sensorMesh)
  })
}
