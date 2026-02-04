<template>
  <div class="reports-content">
    <!-- Reports List View -->
    <div v-if="!selectedReport" class="reports-list-view">
      <div class="reports-header">
        <h2>Available Reports</h2>
        <span class="report-count">{{ reports.length }} reports</span>
      </div>
      
      <div class="reports-list">
        <div 
          v-for="report in reports" 
          :key="report.id" 
          class="report-item"
        >
          <div class="report-icon">
            <i :class="getReportIcon(report.type)"></i>
          </div>
          <div class="report-info">
            <div class="report-title">{{ report.title }}</div>
            <div class="report-meta">
              <span class="report-type">{{ report.type }}</span>
              <span class="report-date">Generated: {{ report.date }}</span>
            </div>
            <div class="report-description">{{ report.description }}</div>
          </div>
          <button class="view-btn" @click="viewReport(report)">
            <i class="fas fa-eye"></i> View
          </button>
        </div>
      </div>
    </div>

    <!-- Report Detail View -->
    <div v-else class="report-detail-view">
      <div class="detail-header">
        <button class="back-btn" @click="goBackToList">
          <i class="fas fa-arrow-left"></i> Back to Reports
        </button>
        <h2>{{ selectedReport.title }}</h2>
        <div class="detail-actions">
          <button class="action-btn">
            <i class="fas fa-download"></i> Download
          </button>
          <button class="action-btn">
            <i class="fas fa-print"></i> Print
          </button>
        </div>
      </div>

      <!-- Line Items Table (existing content) -->
      <div class="line-items-table-section">
        <div class="table-scroll-wrapper">
          <table class="line-items-table">
            <thead>
              <tr>
                <th class="col-line-nbr">Line<br>Nbr</th>
                <th class="col-masl">MASL<br>Description</th>
                <th class="col-qty">Qty</th>
                <th class="col-total-cost">Total<br>Cost</th>
                <th class="col-sc-mos-ta">
                  <div class="header-with-tooltip">
                    SC<br>/(MOS) TA
                  </div>
                </th>
                <th class="col-orc">ORC</th>
                <th class="col-dtc">DTC</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="item in selectedReport.lineItems" :key="item.lineNbr">
                <td>{{ item.lineNbr }}</td>
                <td>{{ item.description }}</td>
                <td>{{ item.qty }}</td>
                <td>{{ item.cost }}</td>
                <td>
                  <div class="sc-dates">
                    <div>{{ item.sc }}</div>
                    <div class="date-range">{{ item.dateRange }}</div>
                    <div>{{ item.ta }}</div>
                  </div>
                </td>
                <td>{{ item.orc }}</td>
                <td>{{ item.dtc }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue';

const selectedReport = ref(null);

// Sample reports data
const reports = ref([
  {
    id: 1,
    title: 'Line Items Summary Report',
    type: 'Financial',
    date: '01/04/2026',
    description: 'Complete listing of all case line items with costs, quantities, and delivery schedules.',
    lineItems: [
      { lineNbr: '001', description: 'NAVAL STRIKE MISSILE, TACTICAL', qty: '96 EA', cost: '$180,000,000', sc: 'X', dateRange: '31MAY2027 - 31MAY2028', ta: 'TA4', orc: 'Z', dtc: '8' },
      { lineNbr: '002', description: 'NAVAL STRIKE MISSILE CONTAINERS', qty: '12 EA', cost: '$1,500,000', sc: 'P', dateRange: '30NOV2026 - 30NOV2027', ta: 'TA5', orc: 'A', dtc: '2' },
      { lineNbr: '003', description: 'NAVAL STRIKE MISSILE GROUND SUPPORT EQUIPMENT', qty: '-', cost: '$8,500,000', sc: 'P', dateRange: '31MAY2027 - 31MAY2028', ta: 'TA5', orc: 'A', dtc: '4' },
      { lineNbr: '004', description: 'TECHNICAL DOCUMENTATION', qty: '-', cost: '$3,200,000', sc: 'P', dateRange: '31MAY2026 - 30NOV2026', ta: 'TA5', orc: 'A', dtc: '4' },
      { lineNbr: '005', description: 'OFFICER TRAINING - NSM Tactical Employment Course', qty: '16 EA', cost: '$720,000', sc: 'S', dateRange: 'SEP2025 - MAR2026', ta: 'TA3', orc: 'A', dtc: '4' },
      { lineNbr: '006', description: 'ENLISTED TRAINING - NSM Maintenance Course', qty: '24 EA', cost: '$672,000', sc: 'S', dateRange: 'OCT2025 - APR2026', ta: 'TA3', orc: 'A', dtc: '4' },
      { lineNbr: '007', description: 'TECHNICAL SERVICES - NSM Integration Support', qty: '-', cost: '$42,500,000', sc: 'S', dateRange: 'MAY2025 - DEC2027', ta: 'TA3', orc: 'A', dtc: '4' },
      { lineNbr: '008', description: 'LOGISTICS SUPPORT SERVICES', qty: '-', cost: '$18,400,000', sc: 'S', dateRange: 'JUN2025 - MAY2028', ta: 'TA3', orc: 'A', dtc: '4' }
    ]
  },
  {
    id: 2,
    title: 'Financial Status Report (FSR)',
    type: 'Financial',
    date: '12/15/2025',
    description: 'Quarterly financial status including obligations, expenditures, and available balances.',
    lineItems: [
      { lineNbr: '001', description: 'NAVAL STRIKE MISSILE, TACTICAL', qty: '96 EA', cost: '$180,000,000', sc: 'X', dateRange: '31MAY2027 - 31MAY2028', ta: 'TA4', orc: 'Z', dtc: '8' },
      { lineNbr: '002', description: 'NAVAL STRIKE MISSILE CONTAINERS', qty: '12 EA', cost: '$1,500,000', sc: 'P', dateRange: '30NOV2026 - 30NOV2027', ta: 'TA5', orc: 'A', dtc: '2' }
    ]
  },
  {
    id: 3,
    title: 'Delivery Schedule Report',
    type: 'Logistics',
    date: '01/02/2026',
    description: 'Current delivery schedule status and projected completion dates for all line items.',
    lineItems: [
      { lineNbr: '001', description: 'NAVAL STRIKE MISSILE, TACTICAL', qty: '96 EA', cost: '$180,000,000', sc: 'X', dateRange: '31MAY2027 - 31MAY2028', ta: 'TA4', orc: 'Z', dtc: '8' }
    ]
  },
  {
    id: 4,
    title: 'Program Management Review (PMR)',
    type: 'Management',
    date: '11/30/2025',
    description: 'Comprehensive program status review including milestones, risks, and action items.',
    lineItems: [
      { lineNbr: '001', description: 'NAVAL STRIKE MISSILE, TACTICAL', qty: '96 EA', cost: '$180,000,000', sc: 'X', dateRange: '31MAY2027 - 31MAY2028', ta: 'TA4', orc: 'Z', dtc: '8' },
      { lineNbr: '002', description: 'NAVAL STRIKE MISSILE CONTAINERS', qty: '12 EA', cost: '$1,500,000', sc: 'P', dateRange: '30NOV2026 - 30NOV2027', ta: 'TA5', orc: 'A', dtc: '2' },
      { lineNbr: '003', description: 'NAVAL STRIKE MISSILE GROUND SUPPORT EQUIPMENT', qty: '-', cost: '$8,500,000', sc: 'P', dateRange: '31MAY2027 - 31MAY2028', ta: 'TA5', orc: 'A', dtc: '4' }
    ]
  },
  {
    id: 5,
    title: 'Training Completion Report',
    type: 'Training',
    date: '12/01/2025',
    description: 'Status of all training activities including completed and scheduled courses.',
    lineItems: [
      { lineNbr: '005', description: 'OFFICER TRAINING - NSM Tactical Employment Course', qty: '16 EA', cost: '$720,000', sc: 'S', dateRange: 'SEP2025 - MAR2026', ta: 'TA3', orc: 'A', dtc: '4' },
      { lineNbr: '006', description: 'ENLISTED TRAINING - NSM Maintenance Course', qty: '24 EA', cost: '$672,000', sc: 'S', dateRange: 'OCT2025 - APR2026', ta: 'TA3', orc: 'A', dtc: '4' }
    ]
  },
  {
    id: 6,
    title: 'Case Closure Checklist',
    type: 'Administrative',
    date: '10/15/2025',
    description: 'Checklist of items required for case closure including outstanding actions.',
    lineItems: []
  }
]);

function getReportIcon(type) {
  const icons = {
    'Financial': 'fas fa-dollar-sign',
    'Logistics': 'fas fa-truck',
    'Management': 'fas fa-tasks',
    'Training': 'fas fa-graduation-cap',
    'Administrative': 'fas fa-file-alt'
  };
  return icons[type] || 'fas fa-file';
}

function viewReport(report) {
  console.log('[ReportsContent] Viewing report:', report.title);
  selectedReport.value = report;
}

function goBackToList() {
  console.log('[ReportsContent] Going back to list');
  selectedReport.value = null;
}
</script>

<style scoped>
.reports-content {
  padding: 0;
  flex-grow: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  background: #f8f9fa;
}

/* ===== REPORTS LIST VIEW ===== */
.reports-list-view {
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow: hidden;
}

.reports-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 20px;
  background: white;
  border-bottom: 1px solid #e5e7eb;
}

.reports-header h2 {
  margin: 0;
  font-size: 1.1rem;
  font-weight: 600;
  color: #2c3e50;
}

.report-count {
  background: #3498db;
  color: white;
  padding: 4px 12px;
  border-radius: 12px;
  font-size: 0.8rem;
  font-weight: 500;
}

.reports-list {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.report-item {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 16px;
  background: white;
  border-radius: 8px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
  transition: box-shadow 0.2s, transform 0.2s;
}

.report-item:hover {
  box-shadow: 0 4px 12px rgba(0,0,0,0.15);
  transform: translateY(-1px);
}

.report-icon {
  width: 48px;
  height: 48px;
  background: linear-gradient(135deg, #3498db, #2980b9);
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.report-icon i {
  color: white;
  font-size: 1.2rem;
}

.report-info {
  flex: 1;
  min-width: 0;
}

.report-title {
  font-weight: 600;
  font-size: 0.95rem;
  color: #2c3e50;
  margin-bottom: 4px;
}

.report-meta {
  display: flex;
  gap: 12px;
  margin-bottom: 4px;
  font-size: 0.75rem;
}

.report-type {
  background: #e8f4fd;
  color: #2980b9;
  padding: 2px 8px;
  border-radius: 4px;
  font-weight: 500;
}

.report-date {
  color: #6b7280;
}

.report-description {
  font-size: 0.8rem;
  color: #6b7280;
  line-height: 1.4;
}

.view-btn {
  padding: 10px 20px;
  background: #3498db;
  color: white;
  border: none;
  border-radius: 6px;
  font-size: 0.85rem;
  font-weight: 500;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 8px;
  transition: background 0.2s;
  flex-shrink: 0;
}

.view-btn:hover {
  background: #2980b9;
}

/* ===== REPORT DETAIL VIEW ===== */
.report-detail-view {
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow: hidden;
}

.detail-header {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 16px 20px;
  background: white;
  border-bottom: 1px solid #e5e7eb;
}

.back-btn {
  padding: 8px 16px;
  background: #f3f4f6;
  color: #374151;
  border: 1px solid #d1d5db;
  border-radius: 6px;
  font-size: 0.85rem;
  font-weight: 500;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 8px;
  transition: all 0.2s;
}

.back-btn:hover {
  background: #e5e7eb;
  border-color: #9ca3af;
}

.detail-header h2 {
  margin: 0;
  font-size: 1.1rem;
  font-weight: 600;
  color: #2c3e50;
  flex: 1;
}

.detail-actions {
  display: flex;
  gap: 8px;
}

.action-btn {
  padding: 8px 14px;
  background: white;
  color: #374151;
  border: 1px solid #d1d5db;
  border-radius: 6px;
  font-size: 0.8rem;
  font-weight: 500;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 6px;
  transition: all 0.2s;
}

.action-btn:hover {
  background: #f9fafb;
  border-color: #9ca3af;
}

/* ===== LINE ITEMS TABLE ===== */
.line-items-table-section {
  flex: 1;
  background: white;
  margin: 16px;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  display: flex;
  flex-direction: column;
}

.table-scroll-wrapper {
  flex: 1;
  overflow: auto;
}

.line-items-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
  table-layout: fixed;
}

.line-items-table thead {
  background: #456379;
  color: white;
  position: sticky;
  top: 0;
  z-index: 10;
}

.line-items-table th {
  padding: 12px 16px;
  text-align: left;
  font-weight: 600;
  border-right: 1px solid rgba(255, 255, 255, 0.2);
  vertical-align: top;
  line-height: 1.3;
}

.line-items-table th:last-child {
  border-right: none;
}

/* Column widths */
.col-line-nbr { width: 6%; }
.col-masl { width: 22%; }
.col-qty { width: 8%; }
.col-total-cost { width: 12%; }
.col-sc-mos-ta { width: 30%; }
.col-orc { width: 7%; }
.col-dtc { width: 7%; }

.header-with-tooltip {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.subheader {
  font-size: 10px;
  font-weight: 400;
  color: #374151;
  font-style: italic;
}

.line-items-table tbody tr {
  border-bottom: 1px solid #e5e7eb;
}

.line-items-table tbody tr:nth-child(odd) {
  background: #f9fafb;
}

.line-items-table tbody tr:hover {
  background: #f3f4f6;
}

.line-items-table td {
  padding: 12px 16px;
  border-right: 1px solid #e5e7eb;
  color: #1a1a1a;
  vertical-align: top;
}

.line-items-table td:last-child {
  border-right: none;
}

.sc-dates {
  display: flex;
  flex-direction: column;
  gap: 2px;
  font-size: 12px;
}

.sc-dates > div:first-child {
  font-weight: 600;
}

.date-range {
  color: #0066cc;
  font-weight: 500;
  font-size: 11px;
}

.sc-dates > div:last-child {
  color: #6b7280;
  font-weight: 600;
}

/* Scrollbar styling */
.reports-list::-webkit-scrollbar,
.table-scroll-wrapper::-webkit-scrollbar {
  width: 8px;
}

.reports-list::-webkit-scrollbar-track,
.table-scroll-wrapper::-webkit-scrollbar-track {
  background: #f1f1f1;
  border-radius: 4px;
}

.reports-list::-webkit-scrollbar-thumb,
.table-scroll-wrapper::-webkit-scrollbar-thumb {
  background: #c1c1c1;
  border-radius: 4px;
}

.reports-list::-webkit-scrollbar-thumb:hover,
.table-scroll-wrapper::-webkit-scrollbar-thumb:hover {
  background: #a1a1a1;
}
</style>