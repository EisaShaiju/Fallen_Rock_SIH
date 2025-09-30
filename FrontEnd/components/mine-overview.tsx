"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import {
  Activity,
  AlertCircle,
  CheckCircle,
  Clock,
  Gauge,
  MapPin,
  Radar,
  Droplets,
  Ruler,
  AsteriskIcon as Seismic,
  CloudRain,
} from "lucide-react"
import { MineViewer3D } from "./mine-viewer-3d"
import { useState, useEffect } from "react"

interface EnhancedSensor {
  id: string
  name: string
  type:
    | "radar"
    | "piezometer"
    | "extensometer"
    | "seismometer"
    | "weather"
    | "inclinometer"
    | "crackmeter"
    | "tiltmeter"
    | "pressure"
    | "temperature"
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
    slope_angle?: number
    tilt?: number
    crack_width?: number
  }
  lastUpdate: string
  alertLevel: "normal" | "warning" | "critical"
}

export function MineOverview() {
  const [hoveredSensor, setHoveredSensor] = useState<EnhancedSensor | null>(null)
  const [sensorFilter, setSensorFilter] = useState<string>("all")

  // State for data, loading, and errors
  // const [sensors, setSensors] = useState<EnhancedSensor[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Fetch data when the component mounts
  // useEffect(() => {
  //   async function fetchSensors() {
  //     try {
  //       // Replace with your actual API endpoint
  //       const response = await fetch("/api/sensors")
  //       if (!response.ok) {
  //         throw new Error("Failed to fetch sensor data from the server.")
  //       }
  //       const data = await response.json()
  //       setSensors(data)
  //     } catch (err: any) {
  //       setError(err.message)
  //     } finally {
  //       setLoading(false)
  //     }
  //   }

  //   fetchSensors()
  // }, []) // Empty array ensures this runs only once

  // // Handle loading and error states
  // if (loading) {
  //   return <div className="p-4">Loading sensor data...</div>
  // }

  // if (error) {
  //   return <div className="p-4 text-red-500">Error: {error}</div>
  // }


  const sensors: EnhancedSensor[] = [
    {
      id: "RDR-001",
      name: "Slope Monitoring Radar",
      type: "radar",
      position: [-15, 8, 10],
      location: "North Wall - Bench 3",
      status: "online",
      readings: { displacement: 2.3, velocity: 0.8 },
      lastUpdate: "30s ago",
      alertLevel: "normal",
    },
    { 
      id: "RDR-002",
      name: "Highwall Radar",
      type: "radar",
      position: [18, 8, -5],
      location: "East Wall - Bench 5",
      status: "warning",
      readings: { displacement: 8.7, velocity: 2.1 },
      lastUpdate: "45s ago",
      alertLevel: "warning",
    },
    {
      id: "PZ-003",
      name: "Groundwater Monitor",
      type: "piezometer",
      position: [-8, 8, 3],
      location: "Pit Floor - Zone A",
      status: "online",
      readings: { pressure: 145.2 },
      lastUpdate: "1m ago",
      alertLevel: "normal",
    },
    {
      id: "PZ-004",
      name: "Pore Pressure Sensor",
      type: "piezometer",
      position: [12, 8, -8],
      location: "South Wall - Bench 2",
      status: "critical",
      readings: { pressure: 198.5 },
      lastUpdate: "2m ago",
      alertLevel: "critical",
    },
    {
      id: "EXT-005",
      name: "Wall Extensometer",
      type: "extensometer",
      position: [-12, 8, -10],
      location: "West Wall - Bench 4",
      status: "online",
      readings: { strain: 0.15 },
      lastUpdate: "15s ago",
      alertLevel: "normal",
    },
    {
      id: "EXT-006",
      name: "Crack Monitor",
      type: "extensometer",
      position: [8, 8, 12],
      location: "North Wall - Bench 6",
      status: "warning",
      readings: { strain: 0.28 },
      lastUpdate: "20s ago",
      alertLevel: "warning",
    },
    {
      id: "SEIS-007",
      name: "Blast Vibration Monitor",
      type: "seismometer",
      position: [0, 8, 0],
      location: "Pit Center - Bottom",
      status: "online",
      readings: { vibration: 12.4 },
      lastUpdate: "10s ago",
      alertLevel: "normal",
    },
    {
      id: "SEIS-008",
      name: "Microseismic Array",
      type: "seismometer",
      position: [-20, 0, 15],
      location: "Perimeter - Station 1",
      status: "online",
      readings: { vibration: 3.2 },
      lastUpdate: "25s ago",
      alertLevel: "normal",
    },
    {
      id: "WX-009",
      name: "Weather Station Alpha",
      type: "weather",
      position: [25, 10, 0],
      location: "Site Office - Roof",
      status: "online",
      readings: { rainfall: 2.1, temperature: 18.5 },
      lastUpdate: "5m ago",
      alertLevel: "normal",
    },
    {
      id: "WX-010",
      name: "Pit Weather Monitor",
      type: "weather",
      position: [0, 10, -20],
      location: "South Rim - Tower",
      status: "offline",
      readings: {},
      lastUpdate: "45m ago",
      alertLevel: "normal",
    },
    {
      id: "INC-011",
      name: "Slope Inclinometer",
      type: "inclinometer",
      position: [-18, 8, 8],
      location: "North Wall - Deep Bore",
      status: "online",
      readings: { slope_angle: 42.3, displacement: 1.8 },
      lastUpdate: "2m ago",
      alertLevel: "normal",
    },
    {
      id: "CM-012",
      name: "Crack Width Monitor",
      type: "crackmeter",
      position: [15, 8, 6],
      location: "East Wall - Tension Crack",
      status: "warning",
      readings: { crack_width: 15.7 },
      lastUpdate: "1m ago",
      alertLevel: "warning",
    },
    {
      id: "TM-013",
      name: "Surface Tiltmeter",
      type: "tiltmeter",
      position: [-10, 8, -3],
      location: "West Rim - Platform",
      status: "online",
      readings: { tilt: 0.034 },
      lastUpdate: "3m ago",
      alertLevel: "normal",
    },
    {
      id: "PT-014",
      name: "Pressure Transducer",
      type: "pressure",
      position: [5, 8, -12],
      location: "Pit Floor - Drainage",
      status: "online",
      readings: { pressure: 87.3 },
      lastUpdate: "90s ago",
      alertLevel: "normal",
    },
    {
      id: "TT-015",
      name: "Temperature Logger",
      type: "temperature",
      position: [-5, 8, 2],
      location: "Pit Floor - Center",
      status: "online",
      readings: { temperature: 22.1 },
      lastUpdate: "5m ago",
      alertLevel: "normal",
    },
  ]

  const filteredSensors = sensorFilter === "all" ? sensors : sensors.filter((sensor) => sensor.type === sensorFilter)

  const stats = {
    totalSensors: sensors.length,
    onlineSensors: sensors.filter((s) => s.status === "online").length,
    warningSensors: sensors.filter((s) => s.status === "warning").length,
    criticalSensors: sensors.filter((s) => s.status === "critical").length,
    activeAlerts: sensors.filter((s) => s.alertLevel !== "normal").length,
    lastSync: "2 minutes ago",
  }

  const sensorTypes = {
    radar: { icon: Radar, label: "Radar", color: "text-blue-500" },
    piezometer: { icon: Droplets, label: "Piezometer", color: "text-cyan-500" },
    extensometer: { icon: Ruler, label: "Extensometer", color: "text-purple-500" },
    seismometer: { icon: Seismic, label: "Seismometer", color: "text-orange-500" },
    weather: { icon: CloudRain, label: "Weather", color: "text-green-500" },
    inclinometer: { icon: Activity, label: "Inclinometer", color: "text-indigo-500" },
    crackmeter: { icon: AlertCircle, label: "Crack Meter", color: "text-red-500" },
    tiltmeter: { icon: Gauge, label: "Tiltmeter", color: "text-yellow-500" },
    pressure: { icon: Droplets, label: "Pressure", color: "text-blue-600" },
    temperature: { icon: Activity, label: "Temperature", color: "text-pink-500" },
  }

  return (
    <div className="space-y-6">
      {/* Enhanced Stats Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Sensors</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.totalSensors}</div>
            <p className="text-xs text-muted-foreground">Monitoring network</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Online</CardTitle>
            <CheckCircle className="h-4 w-4 text-green-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-500">{stats.onlineSensors}</div>
            <p className="text-xs text-muted-foreground">
              {Math.round((stats.onlineSensors / stats.totalSensors) * 100)}% operational
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Warning</CardTitle>
            <AlertCircle className="h-4 w-4 text-yellow-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-yellow-500">{stats.warningSensors}</div>
            <p className="text-xs text-muted-foreground">Need attention</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Critical</CardTitle>
            <AlertCircle className="h-4 w-4 text-red-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-red-500">{stats.criticalSensors}</div>
            <p className="text-xs text-muted-foreground">Urgent action</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Last Sync</CardTitle>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.lastSync}</div>
            <p className="text-xs text-muted-foreground">Data refresh</p>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-4 gap-6">
        {/* Enhanced 3D Mine Model */}
        <Card className="xl:col-span-3">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <MapPin className="h-5 w-5" />
              3D Mine Model
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="aspect-video">
              <MineViewer3D sensors={filteredSensors} onSensorHover={setHoveredSensor} />
            </div>

            {/* Enhanced sensor overlay with type indicators */}
            <div className="mt-4 space-y-3">
              <div className="flex items-center justify-between">
                <h4 className="text-sm font-medium">Sensor Network</h4>
                <Select value={sensorFilter} onValueChange={setSensorFilter}>
                  <SelectTrigger className="w-40">
                    <SelectValue placeholder="Filter sensors" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Types</SelectItem>
                    <SelectItem value="radar">Radar</SelectItem>
                    <SelectItem value="piezometer">Piezometer</SelectItem>
                    <SelectItem value="extensometer">Extensometer</SelectItem>
                    <SelectItem value="seismometer">Seismometer</SelectItem>
                    <SelectItem value="weather">Weather</SelectItem>
                    <SelectItem value="inclinometer">Inclinometer</SelectItem>
                    <SelectItem value="crackmeter">Crack Meter</SelectItem>
                    <SelectItem value="tiltmeter">Tiltmeter</SelectItem>
                    <SelectItem value="pressure">Pressure</SelectItem>
                    <SelectItem value="temperature">Temperature</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="flex flex-wrap gap-2">
                {filteredSensors.map((sensor) => {
                  const TypeIcon = sensorTypes[sensor.type].icon
                  return (
                    <div
                      key={sensor.id}
                      className={`flex items-center gap-2 px-3 py-1 rounded-full text-xs transition-colors border ${
                        hoveredSensor?.id === sensor.id
                          ? "bg-primary text-primary-foreground border-primary"
                          : "bg-muted text-muted-foreground border-border"
                      }`}
                    >
                      <TypeIcon className="h-3 w-3" />
                      <div
                        className={`h-2 w-2 rounded-full ${
                          sensor.status === "online"
                            ? "bg-green-500"
                            : sensor.status === "warning"
                              ? "bg-yellow-500"
                              : sensor.status === "critical"
                                ? "bg-red-500"
                                : "bg-gray-500"
                        }`}
                      />
                      {sensor.id}
                    </div>
                  )
                })}
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Enhanced Sensor List */}
        <Card className="xl:col-span-1">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Gauge className="h-5 w-5" />
              Sensor Details
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 h-96 overflow-y-auto">
            {filteredSensors.map((sensor) => {
              const TypeIcon = sensorTypes[sensor.type].icon
              return (
                <div
                  key={sensor.id}
                  className={`space-y-2 p-3 rounded-lg border transition-colors ${
                    hoveredSensor?.id === sensor.id ? "border-primary bg-primary/5" : "border-border"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <TypeIcon className={`h-4 w-4 ${sensorTypes[sensor.type].color}`} />
                      <Badge
                        variant={
                          sensor.status === "online"
                            ? "default"
                            : sensor.status === "warning"
                              ? "secondary"
                              : sensor.status === "critical"
                                ? "destructive"
                                : "outline"
                        }
                      >
                        {sensor.status}
                      </Badge>
                      <span className="text-sm font-medium">{sensor.id}</span>
                    </div>
                    <span className="text-xs text-muted-foreground">{sensor.lastUpdate}</span>
                  </div>

                  <div className="space-y-1">
                    <p className="text-xs font-medium text-muted-foreground">{sensorTypes[sensor.type].label}</p>
                    <p className="text-xs text-muted-foreground">{sensor.location}</p>
                  </div>

                  {sensor.status !== "offline" && Object.keys(sensor.readings).length > 0 && (
                    <div className="grid grid-cols-1 gap-1 text-xs">
                      {sensor.readings.displacement !== undefined && (
                        <div className="flex justify-between">
                          <span>Displacement:</span>
                          <span className="font-medium">{sensor.readings.displacement}mm</span>
                        </div>
                      )}
                      {sensor.readings.velocity !== undefined && (
                        <div className="flex justify-between">
                          <span>Velocity:</span>
                          <span className="font-medium">{sensor.readings.velocity}mm/day</span>
                        </div>
                      )}
                      {sensor.readings.pressure !== undefined && (
                        <div className="flex justify-between">
                          <span>Pressure:</span>
                          <span className="font-medium">{sensor.readings.pressure}kPa</span>
                        </div>
                      )}
                      {sensor.readings.strain !== undefined && (
                        <div className="flex justify-between">
                          <span>Strain:</span>
                          <span className="font-medium">{sensor.readings.strain}%</span>
                        </div>
                      )}
                      {sensor.readings.vibration !== undefined && (
                        <div className="flex justify-between">
                          <span>Vibration:</span>
                          <span className="font-medium">{sensor.readings.vibration}mm/s²</span>
                        </div>
                      )}
                      {sensor.readings.rainfall !== undefined && (
                        <div className="flex justify-between">
                          <span>Rainfall:</span>
                          <span className="font-medium">{sensor.readings.rainfall}mm</span>
                        </div>
                      )}
                      {sensor.readings.temperature !== undefined && (
                        <div className="flex justify-between">
                          <span>Temperature:</span>
                          <span className="font-medium">{sensor.readings.temperature}°C</span>
                        </div>
                      )}
                      {sensor.readings.slope_angle !== undefined && (
                        <div className="flex justify-between">
                          <span>Slope Angle:</span>
                          <span className="font-medium">{sensor.readings.slope_angle}°</span>
                        </div>
                      )}
                      {sensor.readings.tilt !== undefined && (
                        <div className="flex justify-between">
                          <span>Tilt:</span>
                          <span className="font-medium">{sensor.readings.tilt}°</span>
                        </div>
                      )}
                      {sensor.readings.crack_width !== undefined && (
                        <div className="flex justify-between">
                          <span>Crack Width:</span>
                          <span className="font-medium">{sensor.readings.crack_width}mm</span>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )
            })}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
