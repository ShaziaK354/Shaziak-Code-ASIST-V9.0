<template>
  <div class="portfolio-dashboard">
    <!-- Header Section -->
    <header class="dashboard-header">
      <div class="header-content">
        <h1 class="page-title">Portfolio Overview</h1>
        <p class="page-subtitle">FMS Case Management Dashboard</p>
      </div>
      <div class="header-actions">
        <button class="action-btn secondary">
          <i class="fas fa-download"></i>
          Export Report
        </button>
        <button class="action-btn primary">
          <i class="fas fa-plus"></i>
          New Case
        </button>
      </div>
    </header>

    <!-- KPI Cards -->
    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-icon purple"><i class="fas fa-globe-americas"></i></div>
        <div class="kpi-content">
          <span class="kpi-value">{{ kpiData.totalCountries }}</span>
          <span class="kpi-label">Total Countries</span>
        </div>
      </div>

      <div class="kpi-card">
        <div class="kpi-icon blue"><i class="fas fa-folder-open"></i></div>
        <div class="kpi-content">
          <span class="kpi-value">{{ kpiData.totalCases }}</span>
          <span class="kpi-label">Total Cases</span>
        </div>
      </div>

      <div class="kpi-card">
        <div class="kpi-icon teal"><i class="fas fa-tasks"></i></div>
        <div class="kpi-content">
          <span class="kpi-value">{{ kpiData.caseDevActions }}</span>
          <span class="kpi-label">Case Dev Actions</span>
        </div>
      </div>

      <div class="kpi-card">
        <div class="kpi-icon green"><i class="fas fa-dollar-sign"></i></div>
        <div class="kpi-content">
          <span class="kpi-value">${{ formatCurrency(kpiData.totalCaseValue) }}</span>
          <span class="kpi-label">Total Case Value</span>
        </div>
      </div>

      <div class="kpi-card">
        <div class="kpi-icon cyan"><i class="fas fa-hand-holding-usd"></i></div>
        <div class="kpi-content">
          <span class="kpi-value">${{ formatCurrency(kpiData.totalOAReceived) }}</span>
          <span class="kpi-label">Total OA Received</span>
        </div>
      </div>

      <div class="kpi-card">
        <div class="kpi-icon coral"><i class="fas fa-file-invoice-dollar"></i></div>
        <div class="kpi-content">
          <span class="kpi-value">${{ formatCurrency(kpiData.grossObligations) }}</span>
          <span class="kpi-label">Gross Obligations</span>
        </div>
      </div>

      <div class="kpi-card">
        <div class="kpi-icon pink"><i class="fas fa-credit-card"></i></div>
        <div class="kpi-content">
          <span class="kpi-value">${{ formatCurrency(kpiData.expended) }}</span>
          <span class="kpi-label">Expended</span>
        </div>
      </div>

      <div class="kpi-card">
        <div class="kpi-icon indigo"><i class="fas fa-clock"></i></div>
        <div class="kpi-content">
          <span class="kpi-value">${{ formatCurrency(kpiData.totalULOs) }}</span>
          <span class="kpi-label">Total ULOs</span>
        </div>
      </div>

      <div class="kpi-card highlight">
        <div class="kpi-icon white"><i class="fas fa-wallet"></i></div>
        <div class="kpi-content">
          <span class="kpi-value">${{ formatCurrency(kpiData.availableFunding) }}</span>
          <span class="kpi-label">Available Funding</span>
        </div>
      </div>
    </div>

    <!-- Charts Section -->
    <div class="charts-grid">
      <!-- Case Statuses Chart -->
      <div class="chart-card">
        <div class="chart-header">
          <h3 class="chart-title">Case Statuses</h3>
          <div class="chart-legend">
            <span class="legend-item"><span class="dot blue"></span>Active</span>
            <span class="legend-item"><span class="dot purple"></span>Complete</span>
            <span class="legend-item"><span class="dot gray"></span>Closed</span>
          </div>
        </div>
        <div class="bar-chart">
          <div class="bar-row">
            <span class="bar-label">Active</span>
            <div class="bar-track">
              <div class="bar-fill blue" :style="{ width: getBarWidth(caseStatuses.active, maxCaseStatus) + '%' }">
                <span class="bar-value">{{ caseStatuses.active }}</span>
              </div>
            </div>
          </div>
          <div class="bar-row">
            <span class="bar-label">Supply Complete</span>
            <div class="bar-track">
              <div class="bar-fill purple" :style="{ width: getBarWidth(caseStatuses.supplyServicesComplete, maxCaseStatus) + '%' }">
                <span class="bar-value">{{ caseStatuses.supplyServicesComplete }}</span>
              </div>
            </div>
          </div>
          <div class="bar-row">
            <span class="bar-label">Interim Closed</span>
            <div class="bar-track">
              <div class="bar-fill gray" :style="{ width: getBarWidth(caseStatuses.interimClosed, maxCaseStatus) + '%' }">
                <span class="bar-value">{{ caseStatuses.interimClosed }}</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Funding Document Status -->
      <div class="chart-card">
        <div class="chart-header">
          <h3 class="chart-title">Funding Document Status</h3>
        </div>
        <div class="donut-wrapper">
          <svg class="donut-svg" viewBox="0 0 200 200">
            <circle cx="100" cy="100" r="70" fill="none" stroke="#e5e7eb" stroke-width="28"/>
            <circle cx="100" cy="100" r="70" fill="none" stroke="#1e3a5f" stroke-width="28"
              :stroke-dasharray="getDonutSegment(fundingStatus.active, fundingTotal)"
              stroke-dashoffset="0" transform="rotate(-90 100 100)"/>
            <circle cx="100" cy="100" r="70" fill="none" stroke="#f59e0b" stroke-width="28"
              :stroke-dasharray="getDonutSegment(fundingStatus.pending, fundingTotal)"
              :stroke-dashoffset="getDonutOffset(fundingStatus.active, fundingTotal)"
              transform="rotate(-90 100 100)"/>
            <circle cx="100" cy="100" r="70" fill="none" stroke="#ef4444" stroke-width="28"
              :stroke-dasharray="getDonutSegment(fundingStatus.expired, fundingTotal)"
              :stroke-dashoffset="getDonutOffset(fundingStatus.active + fundingStatus.pending, fundingTotal)"
              transform="rotate(-90 100 100)"/>
            <text x="100" y="95" text-anchor="middle" class="donut-value">{{ fundingTotal }}</text>
            <text x="100" y="115" text-anchor="middle" class="donut-label">Total</text>
          </svg>
          <div class="donut-legend">
            <span><span class="dot blue"></span>Active ({{ fundingStatus.active }})</span>
            <span><span class="dot yellow"></span>Pending ({{ fundingStatus.pending }})</span>
            <span><span class="dot red"></span>Expired ({{ fundingStatus.expired }})</span>
          </div>
        </div>
      </div>

      <!-- CEPT Metrics -->
      <div class="chart-card">
        <div class="chart-header">
          <h3 class="chart-title">CEPT Metrics</h3>
          <span class="badge">Performance</span>
        </div>
        <table class="metrics-table">
          <thead>
            <tr>
              <th>Status</th>
              <th>Cases</th>
              <th>%</th>
              <th>Distribution</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><span class="status green"><i class="fas fa-check-circle"></i> GREEN</span></td>
              <td class="num">{{ ceptMetrics.green.cases }}</td>
              <td>{{ ceptMetrics.green.percentage }}%</td>
              <td>
                <div class="mini-track">
                  <div class="mini-fill green" :style="{ width: ceptMetrics.green.percentage + '%' }"></div>
                </div>
              </td>
            </tr>
            <tr>
              <td><span class="status yellow"><i class="fas fa-exclamation-circle"></i> YELLOW</span></td>
              <td class="num">{{ ceptMetrics.yellow.cases }}</td>
              <td>{{ ceptMetrics.yellow.percentage }}%</td>
              <td>
                <div class="mini-track">
                  <div class="mini-fill yellow" :style="{ width: ceptMetrics.yellow.percentage + '%' }"></div>
                </div>
              </td>
            </tr>
            <tr>
              <td><span class="status red"><i class="fas fa-times-circle"></i> RED</span></td>
              <td class="num">{{ ceptMetrics.red.cases }}</td>
              <td>{{ ceptMetrics.red.percentage }}%</td>
              <td>
                <div class="mini-track">
                  <div class="mini-fill red" :style="{ width: ceptMetrics.red.percentage + '%' }"></div>
                </div>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- Alerts Cards Grid -->
    <div class="alerts-section">
      <h3 class="section-title"><i class="fas fa-bell"></i> Alerts & Insights</h3>
      <div class="alerts-grid">
        <div 
          v-for="alert in alerts" 
          :key="alert.id" 
          :class="['alert-card', alert.severity]"
          @click="navigateToCase(alert.caseId)"
        >
          <div class="alert-card-header">
            <span class="alert-date">{{ alert.date }}</span>
            <span :class="['alert-badge', alert.severity]">{{ alert.severity.toUpperCase() }}</span>
            <span class="alert-case-badge">{{ alert.caseId }}</span>
          </div>
          <div class="alert-card-body">
            <p class="alert-text">{{ alert.text }}</p>
          </div>
          <div class="alert-card-footer">
            <a v-if="alert.link" href="#" class="alert-action" @click.stop>{{ alert.linkText }} →</a>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'

const emit = defineEmits(['case-clicked'])

// KPI Data
const kpiData = ref({
  totalCountries: 4,
  totalCases: 4,
  caseDevActions: 4,
  totalCaseValue: 973500000,
  totalOAReceived: 41000000,
  grossObligations: 45630000,
  expended: 9720000,
  totalULOs: 35900000,
  availableFunding: 57000000
})

// Case Statuses
const caseStatuses = ref({
  active: 37,
  supplyServicesComplete: 11,
  interimClosed: 0
})

const maxCaseStatus = computed(() => Math.max(caseStatuses.value.active, caseStatuses.value.supplyServicesComplete, caseStatuses.value.interimClosed, 1))

// Funding Status
const fundingStatus = ref({ active: 87, pending: 10, expired: 3 })
const fundingTotal = computed(() => fundingStatus.value.active + fundingStatus.value.pending + fundingStatus.value.expired)

// CEPT Metrics
const ceptMetrics = ref({
  green: { cases: 45, percentage: 90 },
  yellow: { cases: 3, percentage: 6 },
  red: { cases: 2, percentage: 4 }
})

// Alerts
const alerts = ref([
  { id: 1, date: '9/30/2025', caseId: 'SR-P-NAV', text: 'N0002425WX12345 for NAVSEA Salary expired', link: true, linkText: 'View in N-ERP', severity: 'critical' },
  { id: 2, date: '11/23/2025', caseId: 'AT-P-SUB', text: 'Action items 004, 005, 007, 011, 016, 026 responses are due', link: true, linkText: 'View Report', severity: 'warning' },
  { id: 3, date: '11/27/2025', caseId: 'SR-P-NAV', text: 'YELLOW CEPT violation will appear on Lines 001 & 003', link: true, linkText: 'View Report', severity: 'warning' },
  { id: 4, date: '12/01/2025', caseId: 'SR-P-NAV', text: 'Quarterly financial review due', link: true, linkText: 'Start Review', severity: 'info' },
  { id: 5, date: '12/15/2025', caseId: 'TW-P-MSL', text: 'LOA amendment pending approval', link: true, linkText: 'View Details', severity: 'info' }
])

function navigateToCase(caseId) {
  emit('case-clicked', caseId)
}

// Helper Functions
function formatCurrency(value) {
  if (value >= 1000000000) return (value / 1000000000).toFixed(2) + 'B'
  if (value >= 1000000) return (value / 1000000).toFixed(2) + 'M'
  if (value >= 1000) return (value / 1000).toFixed(2) + 'K'
  return value.toString()
}

function getBarWidth(value, max) {
  return Math.max((value / max) * 100, 5)
}

function getDonutSegment(value, total) {
  const c = 2 * Math.PI * 70
  return `${(value / total) * c} ${c}`
}

function getDonutOffset(consumed, total) {
  const c = 2 * Math.PI * 70
  return -((consumed / total) * c)
}
</script>

<style scoped>
* { box-sizing: border-box; }

.portfolio-dashboard {
  padding: 16px;
  background: #f8fafc;
  min-height: 100%;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  display: flex;
  flex-direction: column;
  gap: 16px;
  width: 100%;
}

/* Header */
.dashboard-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 12px;
}

.header-content { min-width: 0; }
.page-title { font-size: 22px; font-weight: 700; color: #1f2937; margin: 0; }
.page-subtitle { font-size: 13px; color: #6b7280; margin: 4px 0 0; }

.header-actions { display: flex; gap: 10px; flex-wrap: wrap; }
.action-btn {
  padding: 10px 18px;
  border-radius: 8px;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  border: none;
  display: flex;
  align-items: center;
  gap: 8px;
  white-space: nowrap;
}
.action-btn.secondary { background: white; color: #1f2937; border: 1px solid #e5e7eb; }
.action-btn.primary { background: #1e3a5f; color: white; }

/* KPI Grid - Wrap to new rows if needed */
.kpi-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  width: 100%;
}

.kpi-card {
  background: white;
  border-radius: 12px;
  padding: 16px 20px;
  display: flex;
  align-items: center;
  gap: 12px;
  border: 1px solid #e5e7eb;
  min-width: 220px;
  flex: 0 0 auto;
  overflow: visible;
}

.kpi-card.highlight {
  background: linear-gradient(135deg, #10b981, #059669);
  border: none;
  overflow: visible;
}
.kpi-card.highlight .kpi-value,
.kpi-card.highlight .kpi-label { color: white; }

.kpi-icon {
  width: 44px;
  height: 44px;
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 18px;
  flex-shrink: 0;
}
.kpi-icon.purple { background: #ede9fe; color: #8b5cf6; }
.kpi-icon.blue { background: #dbeafe; color: #2563eb; }
.kpi-icon.teal { background: #ccfbf1; color: #14b8a6; }
.kpi-icon.green { background: #d1fae5; color: #10b981; }
.kpi-icon.cyan { background: #cffafe; color: #06b6d4; }
.kpi-icon.coral { background: #ffedd5; color: #f97316; }
.kpi-icon.pink { background: #fce7f3; color: #ec4899; }
.kpi-icon.indigo { background: #e0e7ff; color: #6366f1; }
.kpi-icon.white { background: rgba(255,255,255,0.2); color: white; }

.kpi-content { display: flex; flex-direction: column; flex: 1; }
.kpi-value { font-size: 20px; font-weight: 700; color: #1f2937; white-space: nowrap; }
.kpi-label { font-size: 10px; color: #6b7280; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 2px; white-space: nowrap; }

/* Charts Grid */
.charts-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 16px;
  width: 100%;
}

.chart-card {
  background: white;
  border-radius: 12px;
  padding: 20px;
  border: 1px solid #e5e7eb;
  min-width: 0;
  overflow: hidden;
}

.chart-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
  flex-wrap: wrap;
  gap: 8px;
}
.chart-title { font-size: 15px; font-weight: 600; color: #1f2937; margin: 0; }
.chart-legend { display: flex; gap: 12px; flex-wrap: wrap; }
.legend-item { display: flex; align-items: center; gap: 5px; font-size: 11px; color: #6b7280; }
.dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.dot.blue { background: #1e3a5f; }
.dot.purple { background: #8b5cf6; }
.dot.gray { background: #94a3b8; }
.dot.yellow { background: #f59e0b; }
.dot.green { background: #10b981; }
.dot.red { background: #ef4444; }

/* Bar Chart */
.bar-chart { display: flex; flex-direction: column; gap: 12px; }
.bar-row { display: flex; align-items: center; gap: 10px; }
.bar-label { width: 100px; font-size: 12px; color: #6b7280; text-align: right; flex-shrink: 0; }
.bar-track { flex: 1; height: 32px; background: #f1f5f9; border-radius: 6px; overflow: hidden; min-width: 0; }
.bar-fill { height: 100%; border-radius: 6px; display: flex; align-items: center; justify-content: flex-end; padding-right: 10px; }
.bar-fill.blue { background: linear-gradient(90deg, #1e3a5f, #2d5a8c); }
.bar-fill.purple { background: linear-gradient(90deg, #8b5cf6, #a78bfa); }
.bar-fill.gray { background: linear-gradient(90deg, #64748b, #94a3b8); }
.bar-value { font-size: 13px; font-weight: 600; color: white; }

/* Donut Chart */
.donut-wrapper { display: flex; flex-direction: column; align-items: center; gap: 16px; }
.donut-svg { width: 150px; height: 150px; max-width: 100%; }
.donut-value { font-size: 26px; font-weight: 700; fill: #1f2937; }
.donut-label { font-size: 11px; fill: #6b7280; }
.donut-legend { display: flex; flex-wrap: wrap; justify-content: center; gap: 14px; }
.donut-legend span { display: flex; align-items: center; gap: 5px; font-size: 11px; color: #6b7280; }

/* CEPT Metrics */
.badge { background: #f1f5f9; color: #6b7280; padding: 4px 10px; border-radius: 12px; font-size: 10px; font-weight: 600; text-transform: uppercase; }
.metrics-table { width: 100%; border-collapse: collapse; font-size: 12px; }
.metrics-table th { text-align: left; padding: 10px 6px; font-size: 10px; font-weight: 600; color: #9ca3af; text-transform: uppercase; border-bottom: 1px solid #e5e7eb; }
.metrics-table td { padding: 12px 6px; border-bottom: 1px solid #f1f5f9; }
.status { display: flex; align-items: center; gap: 6px; font-weight: 600; font-size: 12px; }
.status.green { color: #10b981; }
.status.yellow { color: #f59e0b; }
.status.red { color: #ef4444; }
.num { font-size: 18px; font-weight: 700; color: #1f2937; }
.mini-track { height: 6px; background: #f1f5f9; border-radius: 3px; overflow: hidden; min-width: 60px; }
.mini-fill { height: 100%; border-radius: 3px; }
.mini-fill.green { background: #10b981; }
.mini-fill.yellow { background: #f59e0b; }
.mini-fill.red { background: #ef4444; }

/* Alerts Section */
.alerts-section {
  width: 100%;
}

.section-title {
  font-size: 16px;
  font-weight: 600;
  color: #1f2937;
  margin: 0 0 16px 0;
  display: flex;
  align-items: center;
  gap: 8px;
}

.section-title i {
  color: #dc2626;
}

.alerts-grid {
  display: flex;
  flex-direction: column;
  gap: 12px;
  width: 100%;
}

.alert-card {
  background: white;
  border-radius: 12px;
  padding: 16px 20px;
  border: 1px solid #e5e7eb;
  cursor: pointer;
  transition: all 0.2s ease;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  width: 100%;
}

.alert-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.alert-card.critical {
  border-left: 4px solid #dc2626;
}

.alert-card.warning {
  border-left: 4px solid #f59e0b;
}

.alert-card.info {
  border-left: 4px solid #3b82f6;
}

.alert-card-header {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-shrink: 0;
}

.alert-date {
  font-size: 13px;
  color: #6b7280;
  font-weight: 500;
  min-width: 90px;
}

.alert-badge {
  font-size: 10px;
  font-weight: 600;
  padding: 4px 8px;
  border-radius: 12px;
  text-transform: uppercase;
}

.alert-badge.critical {
  background: #fef2f2;
  color: #dc2626;
}

.alert-badge.warning {
  background: #fffbeb;
  color: #d97706;
}

.alert-badge.info {
  background: #eff6ff;
  color: #2563eb;
}

.alert-case-badge {
  font-size: 11px;
  font-weight: 600;
  padding: 4px 10px;
  border-radius: 6px;
  background: #1e3a5f;
  color: white;
  cursor: pointer;
  transition: all 0.2s ease;
}

.alert-case-badge:hover {
  background: #2d5a8c;
  transform: scale(1.05);
}

.alert-card-body {
  flex: 1;
}

.alert-text {
  font-size: 14px;
  color: #1f2937;
  line-height: 1.5;
  margin: 0;
}

.alert-card-footer {
  flex-shrink: 0;
}

.alert-action {
  font-size: 13px;
  font-weight: 600;
  color: #2563eb;
  text-decoration: none;
  transition: color 0.2s;
  white-space: nowrap;
}

.alert-action:hover {
  color: #1d4ed8;
}

/* Responsive */
@media (max-width: 900px) {
  .charts-grid { grid-template-columns: 1fr; }
}

@media (max-width: 600px) {
  .portfolio-dashboard { padding: 12px; }
  .kpi-grid { grid-template-columns: repeat(2, 1fr); }
  .dashboard-header { flex-direction: column; align-items: flex-start; }
}
</style>