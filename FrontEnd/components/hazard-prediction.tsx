"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  AlertTriangle,
  Mountain,
  TrendingDown,
  TrendingUp,
  Layers,
  BarChart3,
  Activity,
  Clock,
  Target,
  Zap,
} from "lucide-react"
// import { MineViewer3D } from "./mine-viewer-3d"
import HtmlFileViewer from './HtmlFileViewer';

interface EnhancedSensor {
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
  alertLevel: "normal" | "warning" | "critical"
}

export function HazardPrediction() {
  const [activeOverlay, setActiveOverlay] = useState<"slope" | "rockfall">("slope")
  const [timeHorizon, setTimeHorizon] = useState<"6h" | "24h" | "7d">("24h")

  // State for data, loading, and errors
  // const [sensors, setSensors] = useState<EnhancedSensor[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Fetch data when the component mounts
  // useEffect(() => {
  //   async function fetchAllData() {
  //     try {
  //       // Fetch all data concurrently
  //       const [sensorsRes, slopeRes, rockfallRes] = await Promise.all([
  //         fetch("/api/sensors"),
  //         fetch("/api/slope-stability"),
  //         fetch("/api/rockfall"),
  //       ])

  //       if (!sensorsRes.ok || !slopeRes.ok || !rockfallRes.ok) {
  //         throw new Error("Failed to fetch all prediction data.")
  //       }

  //       const sensorsData = await sensorsRes.json()
  //       const slopeStabilityData = await slopeRes.json()
  //       const rockfallPredictionData = await rockfallRes.json()

  //       setSensors(sensorsData)
  //       setSlopeData(slopeStabilityData)
  //       setRockfallData(rockfallPredictionData)
  //     } catch (err: any) {
  //       setError(err.message)
  //     } finally {
  //       setLoading(false)
  //     }
  //   }

  //   fetchAllData()
  // }, [])

  // // Handle loading and error states
  // if (loading) {
  //   return <div className="p-4">Loading hazard prediction data...</div>
  // }

  // if (error || !slopeData || !rockfallData) {
  //   return <div className="p-4 text-red-500">Error: {error || "Could not load prediction data."}</div>
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
  ]

  const slopeStabilityData = {
    factorOfSafety: 1.23,
    trend: "decreasing" as const,
    riskLevel: "moderate" as const,
    confidence: 87,
    lastCalculated: "5 minutes ago",
    zones: [
      {
        id: "SZ-001",
        name: "North Wall - Upper Benches",
        fos: 1.45,
        risk: "low" as const,
        area: "2,400 m²",
        volume: "18,000 m³",
        confidence: 92,
      },
      {
        id: "SZ-002",
        name: "East Wall - Critical Section",
        fos: 1.12,
        risk: "moderate" as const,
        area: "1,800 m²",
        volume: "12,600 m³",
        confidence: 78,
      },
      {
        id: "SZ-003",
        name: "South Wall - Weathered Zone",
        fos: 0.89,
        risk: "high" as const,
        area: "3,200 m²",
        volume: "28,800 m³",
        confidence: 85,
      },
      {
        id: "SZ-004",
        name: "West Wall - Stable Section",
        fos: 1.67,
        risk: "low" as const,
        area: "2,100 m²",
        volume: "15,750 m³",
        confidence: 94,
      },
    ],
    forecast: {
      "6h": { fos: 1.21, risk: "moderate" as const },
      "24h": { fos: 1.18, risk: "moderate" as const },
      "7d": { fos: 1.09, risk: "high" as const },
    },
  }

  const rockfallData = {
    probability: 0.34,
    trend: "increasing" as const,
    riskLevel: "moderate" as const,
    confidence: 82,
    lastCalculated: "3 minutes ago",
    zones: [
      {
        id: "RF-001",
        name: "North Face - Loose Rock",
        probability: 0.67,
        risk: "high" as const,
        volume: "450 m³",
        trajectory: "Road access",
        confidence: 76,
      },
      {
        id: "RF-002",
        name: "East Highwall - Overhang",
        probability: 0.23,
        risk: "low" as const,
        volume: "120 m³",
        trajectory: "Bench 2",
        confidence: 89,
      },
      {
        id: "RF-003",
        name: "South Wall - Fractured Zone",
        probability: 0.45,
        risk: "moderate" as const,
        volume: "280 m³",
        trajectory: "Haul road",
        confidence: 71,
      },
      {
        id: "RF-004",
        name: "West Wall - Minor Instability",
        probability: 0.12,
        risk: "low" as const,
        volume: "85 m³",
        trajectory: "Pit floor",
        confidence: 93,
      },
    ],
    forecast: {
      "6h": { probability: 0.31, risk: "moderate" as const },
      "24h": { probability: 0.38, risk: "moderate" as const },
      "7d": { probability: 0.52, risk: "high" as const },
    },
  }

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case "low":
        return "bg-green-500 text-white"
      case "moderate":
        return "bg-yellow-500 text-white"
      case "high":
        return "bg-red-500 text-white"
      default:
        return "bg-gray-500 text-white"
    }
  }

  const getRiskProgress = (risk: string) => {
    switch (risk) {
      case "low":
        return 25
      case "moderate":
        return 60
      case "high":
        return 90
      default:
        return 0
    }
  }

  const currentData = activeOverlay === "slope" ? slopeStabilityData : rockfallData
  const currentForecast = currentData.forecast[timeHorizon]

  return (
    <div className="space-y-6">
      {/* Enhanced Prediction Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Factor of Safety</CardTitle>
            <TrendingDown className="h-4 w-4 text-red-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{slopeStabilityData.factorOfSafety}</div>
            <Progress value={slopeStabilityData.factorOfSafety * 50} className="mt-2" />
            <p className="text-xs text-muted-foreground mt-1">{slopeStabilityData.confidence}% confidence</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Rockfall Risk</CardTitle>
            <TrendingUp className="h-4 w-4 text-red-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{Math.round(rockfallData.probability * 100)}%</div>
            <Progress value={rockfallData.probability * 100} className="mt-2" />
            <p className="text-xs text-muted-foreground mt-1">{rockfallData.confidence}% confidence</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Sensors</CardTitle>
            <Activity className="h-4 w-4 text-green-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{sensors.filter((s) => s.status === "online").length}</div>
            <p className="text-xs text-muted-foreground">of {sensors.length} total sensors</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Last Update</CardTitle>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">Live</div>
            <p className="text-xs text-muted-foreground">Real-time monitoring</p>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-5 gap-6">
        {/* Enhanced 3D Mine Model with Heatmap */}
        <Card className="xl:col-span-3">
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center gap-2">
                <Mountain className="h-5 w-5" />
                Hazard Prediction Model
              </CardTitle>
              <div className="flex gap-2">
                <Button
                  variant={activeOverlay === "slope" ? "default" : "outline"}
                  size="sm"
                  onClick={() => setActiveOverlay("slope")}
                >
                  <Layers className="h-4 w-4 mr-1" />
                  Slope Stability
                </Button>
                <Button
                  variant={activeOverlay === "rockfall" ? "default" : "outline"}
                  size="sm"
                  onClick={() => setActiveOverlay("rockfall")}
                >
                  <AlertTriangle className="h-4 w-4 mr-1" />
                  Rockfall
                </Button>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            {/* <div className="aspect-video">
              <MineViewer3D sensors={sensors} showHeatmap={true} heatmapType={activeOverlay} />
            </div> */}
            <div className="w-full h-[600px]">
              <HtmlFileViewer fileUrl="/minemeshrisk.html" />
            </div>

            {/* Enhanced Heatmap Legend and Controls */}
            <div className="mt-4 space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <div className="flex items-center gap-2">
                    <div className="h-3 w-3 rounded bg-green-500" />
                    <span className="text-xs">Low Risk</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="h-3 w-3 rounded bg-yellow-500" />
                    <span className="text-xs">Moderate Risk</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="h-3 w-3 rounded bg-red-500" />
                    <span className="text-xs">High Risk</span>
                  </div>
                </div>

                <div className="flex gap-1">
                  {(["6h", "24h", "7d"] as const).map((period) => (
                    <Button
                      key={period}
                      variant={timeHorizon === period ? "default" : "outline"}
                      size="sm"
                      onClick={() => setTimeHorizon(period)}
                    >
                      {period}
                    </Button>
                  ))}
                </div>
              </div>

              <div className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground">
                  Forecast ({timeHorizon}): {activeOverlay === "slope" ? "FOS" : "Probability"}
                </span>
                <Badge className={getRiskColor(currentForecast.risk)}>
                  {activeOverlay === "slope"
                    ? (currentForecast as any).fos
                    : `${Math.round((currentForecast as any).probability * 100)}%`}
                </Badge>
              </div>
            </div>

            <div className="mt-6 p-4 rounded-lg bg-muted/30 border border-border">
              <h4 className="text-sm font-medium mb-3 flex items-center gap-2">
                <TrendingDown className="h-4 w-4" />
                Forecast Trends ({timeHorizon})
              </h4>
              <div className="grid grid-cols-3 gap-4">
                {Object.entries(currentData.forecast).map(([period, forecast]) => (
                  <div key={period} className="text-center">
                    <div className="text-xs text-muted-foreground mb-1">{period}</div>
                    <div className="text-lg font-bold">
                      {activeOverlay === "slope"
                        ? (forecast as any).fos
                        : `${Math.round((forecast as any).probability * 100)}%`}
                    </div>
                    <Badge className={getRiskColor(forecast.risk)} variant="outline" size="sm">
                      {forecast.risk}
                    </Badge>
                  </div>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Enhanced Risk Zone Details */}
        <Card className="xl:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5" />
              {activeOverlay === "slope" ? "Stability Analysis" : "Rockfall Analysis"}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <Tabs defaultValue="zones" className="w-full">
              <TabsList className="grid w-full grid-cols-2">
                <TabsTrigger value="zones">Risk Zones</TabsTrigger>
                <TabsTrigger value="forecast">Forecast</TabsTrigger>
              </TabsList>

              <TabsContent value="zones" className="space-y-3 mt-4">
                <div className="max-h-96 overflow-y-auto space-y-3">
                  {currentData.zones.map((zone) => (
                    <div key={zone.id} className="space-y-3 p-3 rounded-lg border border-border">
                      <div className="flex items-center justify-between">
                        <Badge className={getRiskColor(zone.risk)}>{zone.risk} risk</Badge>
                        <span className="text-sm font-medium">{zone.id}</span>
                      </div>

                      <p className="text-xs text-muted-foreground font-medium">{zone.name}</p>

                      <div className="space-y-2">
                        <div className="flex items-center justify-between text-xs">
                          <span className="text-muted-foreground">
                            {activeOverlay === "slope" ? "Factor of Safety:" : "Probability:"}
                          </span>
                          <span className="font-medium">
                            {activeOverlay === "slope"
                              ? (zone as any).fos
                              : `${Math.round((zone as any).probability * 100)}%`}
                          </span>
                        </div>

                        <div className="flex items-center justify-between text-xs">
                          <span className="text-muted-foreground">
                            {activeOverlay === "slope" ? "Volume:" : "Volume:"}
                          </span>
                          <span className="font-medium">
                            {activeOverlay === "slope" ? (zone as any).volume : (zone as any).volume}
                          </span>
                        </div>

                        {activeOverlay === "rockfall" && (
                          <div className="flex items-center justify-between text-xs">
                            <span className="text-muted-foreground">Trajectory:</span>
                            <span className="font-medium">{(zone as any).trajectory}</span>
                          </div>
                        )}

                        <div className="flex items-center justify-between text-xs">
                          <span className="text-muted-foreground">Confidence:</span>
                          <span className="font-medium">{zone.confidence}%</span>
                        </div>

                        <Progress value={getRiskProgress(zone.risk)} className="h-2" />
                      </div>
                    </div>
                  ))}
                </div>
              </TabsContent>

              <TabsContent value="forecast" className="space-y-4 mt-4">
                <div className="space-y-3">
                  {Object.entries(currentData.forecast).map(([period, forecast]) => (
                    <div key={period} className="flex items-center justify-between p-3 rounded-lg border border-border">
                      <div className="flex items-center gap-3">
                        <Target className="h-4 w-4 text-muted-foreground" />
                        <span className="text-sm font-medium">{period}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge className={getRiskColor(forecast.risk)} variant="outline">
                          {activeOverlay === "slope"
                            ? (forecast as any).fos
                            : `${Math.round((forecast as any).probability * 100)}%`}
                        </Badge>
                        <Badge variant="outline">{forecast.risk}</Badge>
                      </div>
                    </div>
                  ))}
                </div>

                <div className="p-3 rounded-lg bg-muted/50 border border-border">
                  <div className="flex items-center gap-2 mb-2">
                    <Zap className="h-4 w-4 text-primary" />
                    <span className="text-sm font-medium">Model Performance</span>
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div className="flex justify-between">
                      <span>Accuracy:</span>
                      <span className="font-medium">{currentData.confidence}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Updated:</span>
                      <span className="font-medium">{currentData.lastCalculated}</span>
                    </div>
                  </div>
                </div>
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
