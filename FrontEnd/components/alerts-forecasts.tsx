"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Progress } from "@/components/ui/progress"
import {
  AlertTriangle,
  Bell,
  TrendingDown,
  Activity,
  Zap,
  Mail,
  MessageSquare,
  CheckCircle,
  XCircle,
  AlertCircle,
  BarChart3,
  Calendar,
  Target,
} from "lucide-react"

interface Alert {
  id: string
  type: "slope_instability" | "rockfall_risk" | "sensor_failure" | "weather_warning" | "vibration_excess"
  severity: "low" | "medium" | "high" | "critical"
  sensorId: string
  sensorLocation: string
  message: string
  timestamp: string
  status: "active" | "acknowledged" | "resolved"
}

interface ForecastData {
  timestamp: string
  factorOfSafety: number
  rockfallProbability: number
  confidence: number
}

const TrendChart = ({ data, type }: { data: ForecastData[]; type: "safety" | "rockfall" }) => {
  const width = 400
  const height = 200
  const padding = 40

  const values = data.map((d) => (type === "safety" ? d.factorOfSafety : d.rockfallProbability * 100))
  const minValue = Math.min(...values)
  const maxValue = Math.max(...values)
  const range = maxValue - minValue || 1

  const points = data.map((d, i) => {
    const x = padding + (i * (width - 2 * padding)) / (data.length - 1)
    const value = type === "safety" ? d.factorOfSafety : d.rockfallProbability * 100
    const y = height - padding - ((value - minValue) / range) * (height - 2 * padding)
    return { x, y, value }
  })

  const pathData = points.map((p, i) => `${i === 0 ? "M" : "L"} ${p.x} ${p.y}`).join(" ")

  const areaData = `M ${points[0].x} ${height - padding} L ${pathData.substring(2)} L ${points[points.length - 1].x} ${height - padding} Z`

  const color = type === "safety" ? "#3b82f6" : "#ef4444"
  const fillColor = type === "safety" ? "#3b82f620" : "#ef444420"

  return (
    <div className="relative">
      <svg width={width} height={height} className="border rounded-lg bg-background">
        {/* Grid lines */}
        {[0, 1, 2, 3, 4].map((i) => (
          <line
            key={i}
            x1={padding}
            y1={padding + (i * (height - 2 * padding)) / 4}
            x2={width - padding}
            y2={padding + (i * (height - 2 * padding)) / 4}
            stroke="#e5e7eb"
            strokeWidth={1}
          />
        ))}

        {/* Area fill */}
        <path d={areaData} fill={fillColor} />

        {/* Trend line */}
        <path d={pathData} stroke={color} strokeWidth={2} fill="none" />

        {/* Data points */}
        {points.map((point, i) => (
          <g key={i}>
            <circle cx={point.x} cy={point.y} r={4} fill={color} />
            <text x={point.x} y={height - 10} textAnchor="middle" className="text-xs fill-muted-foreground">
              {data[i].timestamp}
            </text>
          </g>
        ))}

        {/* Y-axis labels */}
        {[0, 1, 2, 3, 4].map((i) => {
          const value = maxValue - (i * range) / 4
          const displayValue = type === "safety" ? value.toFixed(2) : `${Math.round(value)}%`
          return (
            <text
              key={i}
              x={padding - 10}
              y={padding + (i * (height - 2 * padding)) / 4 + 4}
              textAnchor="end"
              className="text-xs fill-muted-foreground"
            >
              {displayValue}
            </text>
          )
        })}
      </svg>
    </div>
  )
}

export function AlertsForecasts() {
  const [selectedTimeframe, setSelectedTimeframe] = useState<"6h" | "24h" | "7d">("24h")
  const [testAlertSent, setTestAlertSent] = useState(false)

  const alerts: Alert[] = [
    {
      id: "ALT-001",
      type: "slope_instability",
      severity: "high",
      sensorId: "RDR-002",
      sensorLocation: "East Wall - Bench 5",
      message: "Displacement rate exceeding threshold: 2.1 mm/day detected",
      timestamp: "2 minutes ago",
      status: "active",
    },
    {
      id: "ALT-002",
      type: "rockfall_risk",
      severity: "critical",
      sensorId: "EXT-006",
      sensorLocation: "North Wall - Bench 6",
      message: "Critical strain measurement: 0.28% - immediate attention required",
      timestamp: "5 minutes ago",
      status: "acknowledged",
    },
    {
      id: "ALT-003",
      type: "sensor_failure",
      severity: "medium",
      sensorId: "WX-010",
      sensorLocation: "South Rim - Tower",
      message: "Weather station offline for 45 minutes",
      timestamp: "45 minutes ago",
      status: "active",
    },
    {
      id: "ALT-004",
      type: "weather_warning",
      severity: "medium",
      sensorId: "WX-009",
      sensorLocation: "Site Office - Roof",
      message: "Heavy rainfall detected: 15mm in last hour",
      timestamp: "1 hour ago",
      status: "resolved",
    },
    {
      id: "ALT-005",
      type: "vibration_excess",
      severity: "low",
      sensorId: "SEIS-007",
      sensorLocation: "Pit Center - Bottom",
      message: "Blast vibration slightly above normal: 12.4 mm/s²",
      timestamp: "2 hours ago",
      status: "resolved",
    },
  ]

  const forecastData: ForecastData[] = [
    { timestamp: "Now", factorOfSafety: 1.23, rockfallProbability: 0.34, confidence: 87 },
    { timestamp: "6h", factorOfSafety: 1.21, rockfallProbability: 0.31, confidence: 85 },
    { timestamp: "12h", factorOfSafety: 1.19, rockfallProbability: 0.36, confidence: 82 },
    { timestamp: "24h", factorOfSafety: 1.18, rockfallProbability: 0.38, confidence: 78 },
    { timestamp: "48h", factorOfSafety: 1.15, rockfallProbability: 0.42, confidence: 74 },
    { timestamp: "7d", factorOfSafety: 1.09, rockfallProbability: 0.52, confidence: 68 },
  ]

  const riskSummary = {
    next6h: {
      slopeRisk: "moderate",
      rockfallRisk: "low",
      overallRisk: "moderate",
      confidence: 85,
    },
    next24h: {
      slopeRisk: "moderate",
      rockfallRisk: "moderate",
      overallRisk: "moderate",
      confidence: 78, 
    },
  }

  const getAlertIcon = (type: string) => {
    switch (type) {
      case "slope_instability":
        return TrendingDown
      case "rockfall_risk":
        return AlertTriangle
      case "sensor_failure":
        return XCircle
      case "weather_warning":
        return Activity
      case "vibration_excess":
        return Zap
      default:
        return AlertCircle
    }
  }

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case "low":
        return "bg-blue-500 text-white"
      case "medium":
        return "bg-yellow-500 text-white"
      case "high":
        return "bg-orange-500 text-white"
      case "critical":
        return "bg-red-500 text-white"
      default:
        return "bg-gray-500 text-white"
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case "active":
        return "bg-red-100 text-red-800 border-red-200"
      case "acknowledged":
        return "bg-yellow-100 text-yellow-800 border-yellow-200"
      case "resolved":
        return "bg-green-100 text-green-800 border-green-200"
      default:
        return "bg-gray-100 text-gray-800 border-gray-200"
    }
  }

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case "low":
        return "text-green-600"
      case "moderate":
        return "text-yellow-600"
      case "high":
        return "text-red-600"
      default:
        return "text-gray-600"
    }
  }

  const handleSendTestAlert = () => {
    setTestAlertSent(true)
    setTimeout(() => setTestAlertSent(false), 3000)
  }

  const activeAlerts = alerts.filter((alert) => alert.status === "active")
  const acknowledgedAlerts = alerts.filter((alert) => alert.status === "acknowledged")

  return (
    <div className="space-y-6">
      {/* Alert Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Alerts</CardTitle>
            <AlertCircle className="h-4 w-4 text-red-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-red-500">{activeAlerts.length}</div>
            <p className="text-xs text-muted-foreground">Require attention</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Acknowledged</CardTitle>
            <CheckCircle className="h-4 w-4 text-yellow-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-yellow-500">{acknowledgedAlerts.length}</div>
            <p className="text-xs text-muted-foreground">Being addressed</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Risk Level</CardTitle>
            <Target className="h-4 w-4 text-orange-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-orange-500">Moderate</div>
            <p className="text-xs text-muted-foreground">Next 24 hours</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Test Alert</CardTitle>
            <Bell className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <Button
              onClick={handleSendTestAlert}
              disabled={testAlertSent}
              className="w-full"
              variant={testAlertSent ? "secondary" : "default"}
            >
              {testAlertSent ? (
                <>
                  <CheckCircle className="h-4 w-4 mr-2" />
                  Sent!
                </>
              ) : (
                <>
                  <Mail className="h-4 w-4 mr-2" />
                  Send Test
                </>
              )}
            </Button>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Real-time Alert Feed */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Bell className="h-5 w-5" />
              Real-time Alert Feed
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3 max-h-96 overflow-y-auto">
              {alerts.map((alert) => {
                const AlertIcon = getAlertIcon(alert.type)
                return (
                  <div
                    key={alert.id}
                    className={`p-4 rounded-lg border transition-colors ${
                      alert.status === "active" ? "border-red-200 bg-red-50" : "border-border"
                    }`}
                  >
                    <div className="flex items-start gap-3">
                      <AlertIcon className="h-5 w-5 mt-0.5 text-muted-foreground" />
                      <div className="flex-1 space-y-2">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <Badge className={getSeverityColor(alert.severity)}>{alert.severity}</Badge>
                            <Badge variant="outline" className={getStatusColor(alert.status)}>
                              {alert.status}
                            </Badge>
                          </div>
                          <span className="text-xs text-muted-foreground">{alert.timestamp}</span>
                        </div>

                        <p className="text-sm font-medium">{alert.message}</p>

                        <div className="flex items-center justify-between text-xs text-muted-foreground">
                          <span>Sensor: {alert.sensorId}</span>
                          <span>{alert.sensorLocation}</span>
                        </div>

                        {alert.status === "active" && (
                          <div className="flex gap-2 pt-2">
                            <Button size="sm" variant="outline">
                              Acknowledge
                            </Button>
                            <Button size="sm" variant="outline">
                              <MessageSquare className="h-3 w-3 mr-1" />
                              Details
                            </Button>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          </CardContent>
        </Card>

        {/* Risk Summary Widget */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Target className="h-5 w-5" />
              Risk Summary
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-3">
              <div className="p-3 rounded-lg border border-border">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium">Next 6 Hours</span>
                  <Badge variant="outline">{riskSummary.next6h.confidence}% confidence</Badge>
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Slope Risk:</span>
                    <span className={`font-medium ${getRiskColor(riskSummary.next6h.slopeRisk)}`}>
                      {riskSummary.next6h.slopeRisk}
                    </span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span>Rockfall Risk:</span>
                    <span className={`font-medium ${getRiskColor(riskSummary.next6h.rockfallRisk)}`}>
                      {riskSummary.next6h.rockfallRisk}
                    </span>
                  </div>
                  <div className="flex justify-between text-sm font-medium">
                    <span>Overall:</span>
                    <span className={getRiskColor(riskSummary.next6h.overallRisk)}>
                      {riskSummary.next6h.overallRisk}
                    </span>
                  </div>
                </div>
              </div>

              <div className="p-3 rounded-lg border border-border">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium">Next 24 Hours</span>
                  <Badge variant="outline">{riskSummary.next24h.confidence}% confidence</Badge>
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Slope Risk:</span>
                    <span className={`font-medium ${getRiskColor(riskSummary.next24h.slopeRisk)}`}>
                      {riskSummary.next24h.slopeRisk}
                    </span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span>Rockfall Risk:</span>
                    <span className={`font-medium ${getRiskColor(riskSummary.next24h.rockfallRisk)}`}>
                      {riskSummary.next24h.rockfallRisk}
                    </span>
                  </div>
                  <div className="flex justify-between text-sm font-medium">
                    <span>Overall:</span>
                    <span className={getRiskColor(riskSummary.next24h.overallRisk)}>
                      {riskSummary.next24h.overallRisk}
                    </span>
                  </div>
                </div>
              </div>
            </div>

            <div className="pt-3 border-t border-border">
              <div className="flex items-center gap-2 mb-3">
                <Calendar className="h-4 w-4 text-muted-foreground" />
                <span className="text-sm font-medium">Notification Settings</span>
              </div>
              <div className="space-y-2">
                <Button variant="outline" size="sm" className="w-full justify-start bg-transparent">
                  <Mail className="h-3 w-3 mr-2" />
                  Email Alerts: ON
                </Button>
                <Button variant="outline" size="sm" className="w-full justify-start bg-transparent">
                  <MessageSquare className="h-3 w-3 mr-2" />
                  SMS Alerts: ON
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Forecast Graphs */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            Forecast Trends
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="combined" className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="combined">Combined View</TabsTrigger>
              <TabsTrigger value="slope">Factor of Safety</TabsTrigger>
              <TabsTrigger value="rockfall">Rockfall Probability</TabsTrigger>
            </TabsList>

            <TabsContent value="combined" className="space-y-4 mt-6">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
                <div className="space-y-3">
                  <h4 className="text-sm font-medium flex items-center gap-2">
                    <TrendingDown className="h-4 w-4 text-blue-500" />
                    Factor of Safety Trend
                  </h4>
                  <TrendChart data={forecastData} type="safety" />
                </div>

                <div className="space-y-3">
                  <h4 className="text-sm font-medium flex items-center gap-2">
                    <AlertTriangle className="h-4 w-4 text-red-500" />
                    Rockfall Probability Trend
                  </h4>
                  <TrendChart data={forecastData} type="rockfall" />
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-3">
                  <h4 className="text-sm font-medium">Factor of Safety Data</h4>
                  {forecastData.map((data, index) => (
                    <div key={index} className="flex items-center justify-between p-2 rounded border">
                      <span className="text-sm">{data.timestamp}</span>
                      <div className="flex items-center gap-2">
                        <Progress value={data.factorOfSafety * 50} className="w-20" />
                        <span className="text-sm font-medium w-12">{data.factorOfSafety}</span>
                      </div>
                    </div>
                  ))}
                </div>

                <div className="space-y-3">
                  <h4 className="text-sm font-medium">Rockfall Probability Data</h4>
                  {forecastData.map((data, index) => (
                    <div key={index} className="flex items-center justify-between p-2 rounded border">
                      <span className="text-sm">{data.timestamp}</span>
                      <div className="flex items-center gap-2">
                        <Progress value={data.rockfallProbability * 100} className="w-20" />
                        <span className="text-sm font-medium w-12">{Math.round(data.rockfallProbability * 100)}%</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </TabsContent>

            <TabsContent value="slope" className="mt-6">
              <div className="space-y-4">
                <div className="flex justify-center mb-6">
                  <TrendChart data={forecastData} type="safety" />
                </div>

                <div className="grid grid-cols-2 md:grid-cols-6 gap-4">
                  {forecastData.map((data, index) => (
                    <Card key={index}>
                      <CardContent className="p-3 text-center">
                        <div className="text-xs text-muted-foreground mb-1">{data.timestamp}</div>
                        <div className="text-lg font-bold">{data.factorOfSafety}</div>
                        <Progress value={data.factorOfSafety * 50} className="mt-2" />
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </div>
            </TabsContent>

            <TabsContent value="rockfall" className="mt-6">
              <div className="space-y-4">
                <div className="flex justify-center mb-6">
                  <TrendChart data={forecastData} type="rockfall" />
                </div>

                <div className="grid grid-cols-2 md:grid-cols-6 gap-4">
                  {forecastData.map((data, index) => (
                    <Card key={index}>
                      <CardContent className="p-3 text-center">
                        <div className="text-xs text-muted-foreground mb-1">{data.timestamp}</div>
                        <div className="text-lg font-bold">{Math.round(data.rockfallProbability * 100)}%</div>
                        <Progress value={data.rockfallProbability * 100} className="mt-2" />
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  )
}
