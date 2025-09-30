"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { AlertTriangle, CheckCircle, Clock, X } from "lucide-react"

interface Alert {
  id: string
  type: "warning" | "critical" | "info"
  title: string
  message: string
  sensorId?: string
  timestamp: Date
  acknowledged: boolean
}

//Simulated real-time alerts data when page load once

export function RealTimeAlerts() {
  const [alerts, setAlerts] = useState<Alert[]>([])
//     {
//       id: "1",
//       type: "critical", 
//       title: "High Vibration Detected",
//       message: "Sensor S004 reporting vibration levels above threshold (3.1mm/s)",
//       sensorId: "S004",
//       timestamp: new Date(Date.now() - 2 * 60 * 1000),
//       acknowledged: false,
//     },
//     {
//       id: "2",
//       type: "warning",
//       title: "Sensor Offline",
//       message: "Sensor S003 has been offline for 5 minutes",
//       sensorId: "S003",
//       timestamp: new Date(Date.now() - 5 * 60 * 1000),
//       acknowledged: false,
//     },
//     {
//       id: "3",
//       type: "info",
//       title: "Maintenance Scheduled",
//       message: "Routine maintenance scheduled for Tunnel A tomorrow at 08:00",
//       timestamp: new Date(Date.now() - 30 * 60 * 1000),
//       acknowledged: true,
//     },
//   ])

  // Simulate real-time alerts
  useEffect(() => {
    const interval = setInterval(() => {
      // Randomly generate new alerts
      if (Math.random() < 0.25) {
        // 10% chance every 5 seconds
        const newAlert: Alert = {
          id: Date.now().toString(),
          type: Math.random() < 0.3 ? "critical" : Math.random() < 0.6 ? "warning" : "info",
          title: "New Alert Generated",
          message: "This is a simulated real-time alert for demonstration",
          timestamp: new Date(),
          acknowledged: false,
        }

        setAlerts((prev) => [newAlert, ...prev.slice(0, 9)]) // Keep only 10 most recent
      }
    }, 5000)

    return () => clearInterval(interval)
  }, [])

  const acknowledgeAlert = (alertId: string) => {
    setAlerts((prev) => prev.map((alert) => (alert.id === alertId ? { ...alert, acknowledged: true } : alert)))
  }

  const dismissAlert = (alertId: string) => {
    setAlerts((prev) => prev.filter((alert) => alert.id !== alertId))
  }

  const getAlertColor = (type: Alert["type"]) => {
    switch (type) {
      case "critical":
        return "bg-destructive text-destructive-foreground"
      case "warning":
        return "bg-secondary text-secondary-foreground"
      case "info":
        return "bg-primary text-primary-foreground"
      default:
        return "bg-muted text-muted-foreground"
    }
  }

  const getAlertIcon = (type: Alert["type"]) => {
    switch (type) {
      case "critical":
      case "warning":
        return <AlertTriangle className="h-4 w-4" />
      case "info":
        return <CheckCircle className="h-4 w-4" />
      default:
        return <Clock className="h-4 w-4" />
    }
  }

  const unacknowledgedAlerts = alerts.filter((alert) => !alert.acknowledged)

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span className="flex items-center gap-2">
            <AlertTriangle className="h-5 w-5" />
            Real-Time Alerts
          </span>
          {unacknowledgedAlerts.length > 0 && <Badge variant="destructive">{unacknowledgedAlerts.length}</Badge>}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3 max-h-96 overflow-y-auto">
        {alerts.length === 0 ? (
          <div className="text-center py-8 text-muted-foreground">
            <CheckCircle className="h-8 w-8 mx-auto mb-2" />
            <p>No active alerts</p>
          </div>
        ) : (
          alerts.map((alert) => (
            <div
              key={alert.id}
              className={`p-3 rounded-lg border transition-all ${
                alert.acknowledged ? "border-border bg-muted/50 opacity-60" : "border-border bg-card"
              }`}
            >
              <div className="flex items-start justify-between gap-2">
                <div className="flex items-start gap-2 flex-1">
                  <Badge className={getAlertColor(alert.type)} variant="secondary">
                    {getAlertIcon(alert.type)}
                  </Badge>
                  <div className="flex-1 min-w-0">
                    <h4 className="text-sm font-medium">{alert.title}</h4>
                    <p className="text-xs text-muted-foreground mt-1">{alert.message}</p>
                    {alert.sensorId && (
                      <Badge variant="outline" className="mt-1 text-xs">
                        {alert.sensorId}
                      </Badge>
                    )}
                    <p className="text-xs text-muted-foreground mt-1">{alert.timestamp.toLocaleTimeString()}</p>
                  </div>
                </div>

                <div className="flex gap-1">
                  {!alert.acknowledged && (
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => acknowledgeAlert(alert.id)}
                      className="h-6 px-2 text-xs"
                    >
                      Ack
                    </Button>
                  )}
                  <Button size="sm" variant="ghost" onClick={() => dismissAlert(alert.id)} className="h-6 w-6 p-0">
                    <X className="h-3 w-3" />
                  </Button>
                </div>
              </div>
            </div>
          ))
        )}
      </CardContent>
    </Card>
  )
}
