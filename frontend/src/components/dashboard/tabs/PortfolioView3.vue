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
        <div class="kpi-icon cyan"><i class="fas fa-hand-holding-usd"></i></div>
        <div class="kpi-content">
          <span class="kpi-value">${{ formatCurrency(kpiData.totalOAReceived) }}</span>
          <span class="kpi-label">Total OA Received</span>
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
              <td><div class="mini-track"><div class="mini-fill green" :style="{ width: ceptMetrics.green.percentage + '%' }"></div></div></td>
            </tr>
            <tr>
              <td><span class="status yellow"><i class="fas fa-exclamation-circle"></i> YELLOW</span></td>
              <td class="num">{{ ceptMetrics.yellow.cases }}</td>
              <td>{{ ceptMetrics.yellow.percentage }}%</td>
              <td><div class="mini-track"><div class="mini-fill yellow" :style="{ width: ceptMetrics.yellow.percentage + '%' }"></div></div></td>
            </tr>
            <tr>
              <td><span class="status red"><i class="fas fa-times-circle"></i> RED</span></td>
              <td class="num">{{ ceptMetrics.red.cases }}</td>
              <td>{{ ceptMetrics.red.percentage }}%</td>
              <td><div class="mini-track"><div class="mini-fill red" :style="{ width: ceptMetrics.red.percentage + '%' }"></div></div></td>
            </tr>
          </tbody>
        </table>
        <div class="metrics-footer">
          <strong>{{ totalCeptCases }}</strong> Total Cases Evaluated
        </div>
      </div>
    </div>

    <!-- Alerts Section -->
    <div class="alerts-card">
      <div class="alerts-header">
        <h2 class="alerts-title"><i class="fas fa-bell"></i> Active Alerts</h2>
        <div class="alert-tabs">
          <button :class="{ active: alertFilter === 'all' }" @click="alertFilter = 'all'">
            All <span>{{ alerts.length }}</span>
          </button>
          <button class="critical" :class="{ active: alertFilter === 'critical' }" @click="alertFilter = 'critical'">
            Critical <span>{{ criticalCount }}</span>
          </button>
          <button class="warning" :class="{ active: alertFilter === 'warning' }" @click="alertFilter = 'warning'">
            Warning <span>{{ warningCount }}</span>
          </button>
          <button class="info" :class="{ active: alertFilter === 'info' }" @click="alertFilter = 'info'">
            Info <span>{{ infoCount }}</span>
          </button>
        </div>
      </div>

      <div class="alerts-list">
        <div v-for="alert in filteredAlerts" :key="alert.id" class="alert-item" :class="alert.type"
          @click="$emit('case-clicked', alert.caseId)">
          <div class="alert-icon"><i :class="getAlertIcon(alert.type)"></i></div>
          <div class="alert-body">
            <div class="alert-top">
              <span class="alert-case">{{ alert.caseId }}</span>
              <span class="alert-time">{{ alert.time }}</span>
            </div>
            <h4>{{ alert.title }}</h4>
            <p>{{ alert.description }}</p>
            <div class="alert-meta">
              <span><i class="fas fa-flag"></i> {{ alert.country }}</span>
              <span class="tag">{{ alert.category }}</span>
            </div>
          </div>
          <i class="fas fa-chevron-right alert-arrow"></i>
        </div>
      </div>

      <div v-if="filteredAlerts.length === 0" class="no-alerts">
        <i class="fas fa-check-circle"></i>
        <p>No alerts matching your filter</p>
      </div>
    </div>

    <!-- Quick Cases -->
    <div class="cases-grid">
      <div class="case-card" @click="$emit('case-clicked', 'SR-P-NAV')">
        <div class="case-icon blue"><i class="fas fa-ship"></i></div>
        <div class="case-info"><h4>SR-P-NAV</h4><p>Naval Systems - Saudi Arabia</p></div>
        <span class="case-badge green">On Track</span>
      </div>
      <div class="case-card" @click="$emit('case-clicked', 'TW-P-MSL')">
        <div class="case-icon purple"><i class="fas fa-rocket"></i></div>
        <div class="case-info"><h4>TW-P-MSL</h4><p>Missile Defense - Taiwan</p></div>
        <span class="case-badge green">On Track</span>
      </div>
      <div class="case-card" @click="$emit('case-clicked', 'AT-P-SYS')">
        <div class="case-icon teal"><i class="fas fa-plane"></i></div>
        <div class="case-info"><h4>AT-P-SYS</h4><p>Air Traffic - Australia</p></div>
        <span class="case-badge yellow">Attention</span>
      </div>
      <div class="case-card" @click="$emit('case-clicked', 'FR-P-SHP')">
        <div class="case-icon coral"><i class="fas fa-anchor"></i></div>
        <div class="case-info"><h4>FR-P-SHP</h4><p>Ship Building - France</p></div>
        <span class="case-badge red">At Risk</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'

defineEmits(['case-clicked'])

const alertFilter = ref('all')

const kpiData = ref({
  totalCountries: 4,
  totalCases: 4,
  totalOAReceived: 879010000,
  caseDevActions: 4,
  totalCaseValue: 973500000,
  grossObligations: 45630000,
  expended: 9720000,
  totalULOs: 35900000,
  availableFunding: 598270000
})

const caseStatuses = ref({ active: 37, supplyServicesComplete: 11, interimClosed: 2 })
const maxCaseStatus = computed(() => Math.max(caseStatuses.value.active, caseStatuses.value.supplyServicesComplete, caseStatuses.value.interimClosed))

const fundingStatus = ref({ active: 87, pending: 10, expired: 3 })
const fundingTotal = computed(() => fundingStatus.value.active + fundingStatus.value.pending + fundingStatus.value.expired)

const ceptMetrics = ref({
  green: { cases: 45, percentage: 90 },
  yellow: { cases: 3, percentage: 6 },
  red: { cases: 2, percentage: 4 }
})
const totalCeptCases = computed(() => ceptMetrics.value.green.cases + ceptMetrics.value.yellow.cases + ceptMetrics.value.red.cases)

const alerts = ref([
  { id: 1, caseId: 'SR-P-NAV', type: 'critical', title: 'LOA Expiration Imminent', description: 'Letter of Offer and Acceptance expires in 15 days.', country: 'Saudi Arabia', category: 'Financial', time: '2 hours ago' },
  { id: 2, caseId: 'FR-P-SHP', type: 'critical', title: 'Budget Threshold Exceeded', description: 'Case expenditure has exceeded 95% of allocated budget.', country: 'France', category: 'Financial', time: '4 hours ago' },
  { id: 3, caseId: 'TW-P-MSL', type: 'warning', title: 'Delivery Schedule Delay', description: 'Phase 2 delivery milestone at risk. Current delay of 12 days.', country: 'Taiwan', category: 'Logistics', time: '6 hours ago' },
  { id: 4, caseId: 'AT-P-SYS', type: 'warning', title: 'Pending Document Review', description: 'Technical documentation awaiting approval for 8 days.', country: 'Australia', category: 'Documents', time: '1 day ago' },
  { id: 5, caseId: 'SR-P-NAV', type: 'info', title: 'New Amendment Received', description: 'Amendment A003 received and logged.', country: 'Saudi Arabia', category: 'Documents', time: '1 day ago' },
  { id: 6, caseId: 'TW-P-MSL', type: 'info', title: 'Quarterly Report Due', description: 'Q4 progress report submission due in 5 days.', country: 'Taiwan', category: 'Reports', time: '2 days ago' }
])

const criticalCount = computed(() => alerts.value.filter(a => a.type === 'critical').length)
const warningCount = computed(() => alerts.value.filter(a => a.type === 'warning').length)
const infoCount = computed(() => alerts.value.filter(a => a.type === 'info').length)
const filteredAlerts = computed(() => alertFilter.value === 'all' ? alerts.value : alerts.value.filter(a => a.type === alertFilter.value))

function formatCurrency(value) {
  if (value >= 1e9) return (value / 1e9).toFixed(2) + 'B'
  if (value >= 1e6) return (value / 1e6).toFixed(2) + 'M'
  if (value >= 1e3) return (value / 1e3).toFixed(1) + 'K'
  return value.toString()
}

function getBarWidth(value, max) { return Math.max((value / max) * 100, 10) }
function getDonutSegment(value, total) { const c = 2 * Math.PI * 70; return `${(value / total) * c} ${c}` }
function getDonutOffset(consumed, total) { const c = 2 * Math.PI * 70; return -((consumed / total) * c) }
function getAlertIcon(type) {
  const icons = { critical: 'fas fa-exclamation-triangle', warning: 'fas fa-exclamation-circle', info: 'fas fa-info-circle' }
  return icons[type] || 'fas fa-bell'
}
</script>

<style scoped>
* { box-sizing: border-box; }

.portfolio-dashboard {
  padding: 20px;
  background: #f8fafc;
  min-height: 100vh;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Header */
.dashboard-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
  flex-wrap: wrap;
  gap: 12px;
}

.page-title { font-size: 24px; font-weight: 700; color: #1e3a5f; margin: 0; }
.page-subtitle { font-size: 13px; color: #6b7280; margin: 2px 0 0; }
.header-actions { display: flex; gap: 10px; }

.action-btn {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 10px 16px;
  border-radius: 8px;
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  border: none;
  transition: all 0.2s;
}
.action-btn.primary { background: #1e3a5f; color: white; }
.action-btn.primary:hover { background: #2d4a6f; }
.action-btn.secondary { background: white; color: #374151; border: 1px solid #e5e7eb; }
.action-btn.secondary:hover { background: #f9fafb; }

/* KPI Grid */
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
  margin-bottom: 20px;
}

.kpi-card {
  background: white;
  border-radius: 12px;
  padding: 18px;
  display: flex;
  align-items: center;
  gap: 14px;
  border: 1px solid #e5e7eb;
  box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

.kpi-card.highlight {
  background: linear-gradient(135deg, #10b981, #059669);
  border: none;
}
.kpi-card.highlight .kpi-value,
.kpi-card.highlight .kpi-label { color: white; }

.kpi-icon {
  width: 48px;
  height: 48px;
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 20px;
  flex-shrink: 0;
}
.kpi-icon.purple { background: #ede9fe; color: #8b5cf6; }
.kpi-icon.blue { background: #dbeafe; color: #2563eb; }
.kpi-icon.cyan { background: #cffafe; color: #0891b2; }
.kpi-icon.teal { background: #ccfbf1; color: #14b8a6; }
.kpi-icon.green { background: #d1fae5; color: #10b981; }
.kpi-icon.coral { background: #ffedd5; color: #f97316; }
.kpi-icon.pink { background: #fce7f3; color: #ec4899; }
.kpi-icon.indigo { background: #e0e7ff; color: #6366f1; }
.kpi-icon.white { background: rgba(255,255,255,0.2); color: white; }

.kpi-content { display: flex; flex-direction: column; }
.kpi-value { font-size: 22px; font-weight: 700; color: #1f2937; }
.kpi-label { font-size: 11px; color: #6b7280; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 2px; }

/* Charts Grid */
.charts-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 16px;
  margin-bottom: 20px;
}

.chart-card {
  background: white;
  border-radius: 12px;
  padding: 20px;
  border: 1px solid #e5e7eb;
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
.chart-legend { display: flex; gap: 12px; }
.legend-item, .donut-legend span { display: flex; align-items: center; gap: 5px; font-size: 11px; color: #6b7280; }
.dot { width: 8px; height: 8px; border-radius: 50%; }
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
.bar-track { flex: 1; height: 32px; background: #f1f5f9; border-radius: 6px; overflow: hidden; }
.bar-fill {
  height: 100%;
  border-radius: 6px;
  display: flex;
  align-items: center;
  justify-content: flex-end;
  padding-right: 10px;
  transition: width 0.4s;
}
.bar-fill.blue { background: linear-gradient(90deg, #1e3a5f, #2d5a8c); }
.bar-fill.purple { background: linear-gradient(90deg, #8b5cf6, #a78bfa); }
.bar-fill.gray { background: linear-gradient(90deg, #64748b, #94a3b8); }
.bar-value { font-size: 13px; font-weight: 600; color: white; }

/* Donut Chart */
.donut-wrapper { display: flex; flex-direction: column; align-items: center; gap: 16px; }
.donut-svg { width: 150px; height: 150px; }
.donut-value { font-size: 26px; font-weight: 700; fill: #1f2937; }
.donut-label { font-size: 11px; fill: #6b7280; }
.donut-legend { display: flex; flex-wrap: wrap; justify-content: center; gap: 14px; }

/* CEPT Metrics */
.badge { background: #f1f5f9; color: #6b7280; padding: 4px 10px; border-radius: 12px; font-size: 10px; font-weight: 600; text-transform: uppercase; }
.metrics-table { width: 100%; border-collapse: collapse; font-size: 12px; }
.metrics-table th { text-align: left; padding: 10px 6px; font-size: 10px; font-weight: 600; color: #9ca3af; text-transform: uppercase; border-bottom: 1px solid #e5e7eb; }
.metrics-table td { padding: 12px 6px; border-bottom: 1px solid #f1f5f9; }
.metrics-table tr:last-child td { border-bottom: none; }
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
.metrics-footer { text-align: center; padding-top: 12px; margin-top: 12px; border-top: 1px solid #e5e7eb; font-size: 12px; color: #6b7280; }

/* Alerts */
.alerts-card { background: white; border-radius: 12px; padding: 20px; margin-bottom: 20px; border: 1px solid #e5e7eb; }
.alerts-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; flex-wrap: wrap; gap: 12px; }
.alerts-title { font-size: 16px; font-weight: 600; color: #1f2937; margin: 0; display: flex; align-items: center; gap: 8px; }
.alerts-title i { color: #f59e0b; }
.alert-tabs { display: flex; gap: 8px; flex-wrap: wrap; }
.alert-tabs button {
  padding: 6px 12px;
  border-radius: 16px;
  font-size: 12px;
  font-weight: 500;
  cursor: pointer;
  border: 1px solid #e5e7eb;
  background: white;
  color: #6b7280;
  display: flex;
  align-items: center;
  gap: 5px;
  transition: all 0.2s;
}
.alert-tabs button span { background: #f1f5f9; padding: 2px 6px; border-radius: 8px; font-size: 10px; }
.alert-tabs button.active { background: #1e3a5f; color: white; border-color: #1e3a5f; }
.alert-tabs button.active span { background: rgba(255,255,255,0.2); }
.alert-tabs button.critical.active { background: #ef4444; border-color: #ef4444; }
.alert-tabs button.warning.active { background: #f59e0b; border-color: #f59e0b; }
.alert-tabs button.info.active { background: #14b8a6; border-color: #14b8a6; }

.alerts-list { display: flex; flex-direction: column; gap: 10px; }
.alert-item {
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 14px;
  border-radius: 10px;
  cursor: pointer;
  transition: all 0.2s;
  border-left: 4px solid transparent;
}
.alert-item.critical { background: #fef2f2; border-left-color: #ef4444; }
.alert-item.warning { background: #fffbeb; border-left-color: #f59e0b; }
.alert-item.info { background: #f0fdfa; border-left-color: #14b8a6; }
.alert-item:hover { transform: translateX(4px); }
.alert-icon { width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 16px; flex-shrink: 0; color: white; }
.alert-item.critical .alert-icon { background: #ef4444; }
.alert-item.warning .alert-icon { background: #f59e0b; }
.alert-item.info .alert-icon { background: #14b8a6; }
.alert-body { flex: 1; min-width: 0; }
.alert-top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; }
.alert-case { font-size: 11px; font-weight: 600; color: #1e3a5f; background: rgba(30,58,95,0.1); padding: 2px 8px; border-radius: 4px; }
.alert-time { font-size: 11px; color: #9ca3af; }
.alert-body h4 { font-size: 14px; font-weight: 600; color: #1f2937; margin: 0 0 4px; }
.alert-body p { font-size: 12px; color: #6b7280; margin: 0 0 6px; line-height: 1.4; }
.alert-meta { display: flex; gap: 12px; font-size: 11px; color: #9ca3af; }
.alert-meta .tag { background: rgba(0,0,0,0.05); padding: 2px 8px; border-radius: 4px; }
.alert-arrow { color: #9ca3af; font-size: 14px; }
.no-alerts { text-align: center; padding: 30px; color: #9ca3af; }
.no-alerts i { font-size: 36px; color: #10b981; display: block; margin-bottom: 10px; }

/* Cases Grid */
.cases-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; }
.case-card {
  background: white;
  border-radius: 10px;
  padding: 16px;
  display: flex;
  align-items: center;
  gap: 12px;
  border: 1px solid #e5e7eb;
  cursor: pointer;
  transition: all 0.2s;
}
.case-card:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.08); }
.case-icon { width: 42px; height: 42px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 18px; flex-shrink: 0; }
.case-icon.blue { background: #dbeafe; color: #2563eb; }
.case-icon.purple { background: #ede9fe; color: #8b5cf6; }
.case-icon.teal { background: #ccfbf1; color: #14b8a6; }
.case-icon.coral { background: #ffedd5; color: #f97316; }
.case-info { flex: 1; min-width: 0; }
.case-info h4 { font-size: 13px; font-weight: 600; color: #1f2937; margin: 0; }
.case-info p { font-size: 11px; color: #6b7280; margin: 2px 0 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.case-badge { font-size: 10px; font-weight: 600; padding: 4px 10px; border-radius: 12px; white-space: nowrap; }
.case-badge.green { background: #d1fae5; color: #047857; }
.case-badge.yellow { background: #fef3c7; color: #b45309; }
.case-badge.red { background: #fee2e2; color: #b91c1c; }

/* Responsive */
@media (max-width: 1200px) {
  .kpi-grid { grid-template-columns: repeat(4, 1fr); }
  .charts-grid { grid-template-columns: 1fr 1fr; }
  .charts-grid .chart-card:last-child { grid-column: span 2; }
  .cases-grid { grid-template-columns: repeat(2, 1fr); }
}

@media (max-width: 900px) {
  .kpi-grid { grid-template-columns: repeat(2, 1fr); }
  .charts-grid { grid-template-columns: 1fr; }
  .charts-grid .chart-card:last-child { grid-column: span 1; }
}

@media (max-width: 600px) {
  .portfolio-dashboard { padding: 12px; }
  .kpi-grid, .cases-grid { grid-template-columns: 1fr; }
}
</style>