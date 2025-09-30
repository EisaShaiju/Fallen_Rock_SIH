import * as THREE from "three"
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js"

export function loadMineModel(
  scene: THREE.Scene,
  modelPath: string,
  setIsModelLoading: (loading: boolean) => void,
  setModelError: (error: string | null) => void,
) {
  setIsModelLoading(true)
  setModelError(null)

  // Clear existing mine structure
  const existingMine = scene.getObjectByName("mine-structure")
  if (existingMine) {
    scene.remove(existingMine)
  }

  const loader = new GLTFLoader()
  
  loader.load(
    modelPath,
    (gltf) => {
      const model = gltf.scene
      model.name = "mine-structure"
      
      // Scale and position the model appropriately
      model.scale.setScalar(0.1) // Adjust scale as needed
      model.position.set(0, -5, 0) // Adjust position as needed
      
      // Enable shadows
      model.traverse((child) => {
        if (child instanceof THREE.Mesh) {
          child.castShadow = true
          child.receiveShadow = true
        }
      })
      
      scene.add(model)
      setIsModelLoading(false)
    },
    (progress) => {
      // Loading progress
      console.log("Loading progress:", (progress.loaded / progress.total) * 100 + "%")
    },
    (error) => {
      console.error("Error loading model:", error)
      setModelError(`Failed to load model: ${error.message}`)
      setIsModelLoading(false)
      
      // Fallback to procedural model
      createProceduralMineStructure(scene)
    }
  )
}

export function loadBinghamCanyonMine(
  scene: THREE.Scene,
  setIsModelLoading: (loading: boolean) => void,
  setModelError: (error: string | null) => void,
) {
  // Use the Bingham Canyon model by default, This is where you could upload your own model
  const modelPath = "/the-bingham-canyon-mine-utah/source/800004c9-0001-f500-b63f-84710c7967bb.glb"
  loadMineModel(scene, modelPath, setIsModelLoading, setModelError)
}

function createProceduralMineStructure(scene: THREE.Scene) {
  const mineGroup = new THREE.Group()
  mineGroup.name = "mine-structure"

  // Create basic mine structure
  const tunnelGeometry = new THREE.BoxGeometry(12, 3, 3)
  const tunnelMaterial = new THREE.MeshLambertMaterial({ color: 0x8b7355 })
  const tunnel = new THREE.Mesh(tunnelGeometry, tunnelMaterial)
  tunnel.position.set(0, 0, 0)
  tunnel.castShadow = true
  tunnel.receiveShadow = true
  mineGroup.add(tunnel)

  scene.add(mineGroup)
}
