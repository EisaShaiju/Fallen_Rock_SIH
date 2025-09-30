"use client"

import type React from "react"

import { useState } from "react"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { RealTimeAlerts } from "@/components/real-time-alerts"
import { Mountain, AlertTriangle, Menu, X, Bell } from "lucide-react"

interface DashboardLayoutProps {
  children: React.ReactNode
}

export function DashboardLayout({ children }: DashboardLayoutProps) {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [showAlerts, setShowAlerts] = useState(false)
  const pathname = usePathname()

  const navigation = [
    {
      name: "Mine Overview",
      href: "/",
      icon: Mountain,
      current: pathname === "/",
    },
    {
      name: "Hazard Prediction",
      href: "/hazard-prediction",
      icon: AlertTriangle,
      current: pathname === "/hazard-prediction",
    },
    {
      name: "Alerts & Forecasts",
      href: "/alerts-forecasts",
      icon: Bell,
      current: pathname === "/alerts-forecasts",
    },
  ]

  return (
    <div className="min-h-screen gradient-bg">
      {/* Mobile sidebar overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/50 lg:hidden animate-fade-in"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <div
        className={cn(
          "fixed inset-y-0 left-0 z-50 w-64 transform bg-sidebar border-r border-sidebar-border transition-transform duration-300 ease-in-out lg:translate-x-0 animate-slide-up",
          sidebarOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0",
        )}
      >
        <div className="flex h-full flex-col">
          {/* Logo */}
          <div className="flex h-16 items-center justify-between px-6 border-b border-sidebar-border">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary shadow-lg">
                <Mountain className="h-6 w-6 text-primary-foreground" />
              </div>
              <div>
                <span className="text-lg font-bold text-sidebar-foreground">Mine Safety</span>
                <p className="text-xs text-sidebar-foreground/70">Monitoring System</p>
              </div>
            </div>
            <Button variant="ghost" size="sm" className="lg:hidden" onClick={() => setSidebarOpen(false)}>
              <X className="h-4 w-4" />
            </Button>
          </div>

          {/* Navigation */}
          <nav className="flex-1 px-4 py-6 space-y-2">
            {navigation.map((item) => {
              const Icon = item.icon
              return (
                <Link
                  key={item.name}
                  href={item.href}
                  className={cn(
                    "flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
                    item.current
                      ? "bg-sidebar-accent text-sidebar-accent-foreground shadow-sm"
                      : "text-sidebar-foreground hover:bg-sidebar-primary/10 hover:text-sidebar-primary-foreground",
                  )}
                >
                  <Icon className="h-5 w-5" />
                  {item.name}
                </Link>
              )
            })}
          </nav>

          {/* Enhanced System Status */}
          <div className="p-4 border-t border-sidebar-border">
            <Card className="p-4 animate-scale-in">
              <div className="flex items-center gap-2 mb-3">
                <div className="h-2 w-2 rounded-full bg-success animate-pulse" />
                <span className="text-sm font-semibold">System Status</span>
              </div>
              <div className="space-y-2 text-xs">
                <div className="flex justify-between items-center">
                  <span className="text-muted-foreground">Sensors Online</span>
                  <span className="font-bold text-success tabular-nums">24/26</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-muted-foreground">Active Alerts</span>
                  <span className="font-bold text-destructive tabular-nums">3</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-muted-foreground">Last Sync</span>
                  <span className="font-medium text-foreground">2 min ago</span>
                </div>
              </div>
            </Card>
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="lg:pl-64">
        {/* Enhanced Top bar */}
        <div className="sticky top-0 z-30 flex h-16 items-center gap-4 border-b border-border glass-effect px-6">
          <Button variant="ghost" size="sm" className="lg:hidden" onClick={() => setSidebarOpen(true)}>
            <Menu className="h-4 w-4" />
          </Button>

          <div className="flex-1">
            <h1 className="text-xl font-bold text-foreground">
              {navigation.find((item) => item.current)?.name || "Dashboard"}
            </h1>
          </div>

          {/* Enhanced status indicators and alerts */}
          <div className="flex items-center gap-4">
            <Button variant="ghost" size="sm" onClick={() => setShowAlerts(!showAlerts)} className="relative">
              <Bell className="h-4 w-4" />
              <span className="absolute -top-1 -right-1 h-3 w-3 bg-destructive rounded-full animate-pulse shadow-lg" />
            </Button>

            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-success/10 border border-success/20">
              <div className="h-2 w-2 rounded-full bg-success animate-pulse" />
              <span className="text-sm font-medium text-success">Live</span>
            </div>
          </div>
        </div>

        {/* Enhanced Alerts panel */}
        {showAlerts && (
          <div className="border-b border-border glass-effect p-4 animate-slide-up">
            <RealTimeAlerts />
          </div>
        )}

        {/* Page content */}
        <main className="p-6 animate-fade-in">{children}</main>
      </div>
    </div>
  )
}
