<template>
  <div class="tab-content-section-wrapper">
    <!-- Alerts & Insights Section (moved to top) -->
    <div class="alerts-section">
      <div class="alerts-header">
        Alerts & Insights (sorted by due dates)
      </div>
      <div class="alerts-list">
        <div class="alert-item">
          <span class="alert-bullet">➤</span>
          <div class="alert-content">
            <strong>9/30/2025:</strong> N0002425WX12345 for NAVSEA Salary expired
            <a href="#" class="alert-link">(N-ERP)</a> - Consider pulling back additional funding
          </div>
        </div>

        <div class="alert-item">
          <span class="alert-bullet">➤</span>
          <div class="alert-content">
            <strong>11/23/2025:</strong> Action items 004, 005, 007, 011, 016, 026 responses are due
            <a href="#" class="alert-link">(View Report)</a>
          </div>
        </div>

        <div class="alert-item">
          <span class="alert-bullet">➤</span>
          <div class="alert-content">
            <strong>11/27/2025: <span class="yellow-highlight">YELLOW</span></strong> CEPT violation will appear on Lines 001 & 003 - CP-COM
            <a href="#" class="alert-link">(View report)</a>
          </div>
        </div>

        <div class="alert-item">
          <span class="alert-bullet">➤</span>
          <div class="alert-content">
            <strong>11/30/2025:</strong> ACR is due
            <a href="#" class="alert-link">(SCIP)</a>
          </div>
        </div>

        <div class="alert-item">
          <span class="alert-bullet">➤</span>
          <div class="alert-content">
            <strong>3/31/2026:</strong> Line 005 expires - Consider processing a modification to extend MOS
          </div>
        </div>
      </div>
    </div>

    <!-- Line Items Table (moved below alerts) -->
    <div class="line-items-table-section">
      <!-- Single scrollable container for both header and body -->
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
                  <div class="subheader">(Actual Dates, not Period of Performance (12-24))</div>
                </div>
              </th>
              <th class="col-orc">
                <TermTooltip term="ORC" :strategy="'fixed'" :placement="'top'" />
              </th>
              <th class="col-dtc">
                <TermTooltip term="DTC" :strategy="'fixed'" :placement="'top'" />
              </th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>001</td>
              <td>NAVAL STRIKE MISSILE, TACTICAL</td>
              <td>96 EA</td>
              <td>$180,000,000</td>
              <td>
                <div class="sc-dates">
                  <div>X</div>
                  <div class="date-range">31MAY2027 - 31MAY2028</div>
                  <div>TA4</div>
                </div>
              </td>
              <td>Z</td>
              <td>8</td>
            </tr>
            <tr>
              <td>002</td>
              <td>NAVAL STRIKE MISSILE CONTAINERS</td>
              <td>12 EA</td>
              <td>$1,500,000</td>
              <td>
                <div class="sc-dates">
                  <div>P</div>
                  <div class="date-range">30NOV2026 - 30NOV2027</div>
                  <div>TA5</div>
                </div>
              </td>
              <td>A</td>
              <td>2</td>
            </tr>
            <tr>
              <td>003</td>
              <td>NAVAL STRIKE MISSILE GROUND SUPPORT EQUIPMENT</td>
              <td>-</td>
              <td>$8,500,000</td>
              <td>
                <div class="sc-dates">
                  <div>P</div>
                  <div class="date-range">31MAY2027 - 31MAY2028</div>
                  <div>TA5</div>
                </div>
              </td>
              <td>A</td>
              <td>4</td>
            </tr>
            <tr>
              <td>004</td>
              <td>TECHNICAL DOCUMENTATION</td>
              <td>-</td>
              <td>$3,200,000</td>
              <td>
                <div class="sc-dates">
                  <div>P</div>
                  <div class="date-range">31MAY2026 - 30NOV2026</div>
                  <div>TA5</div>
                </div>
              </td>
              <td>A</td>
              <td>4</td>
            </tr>
            <tr>
              <td>005</td>
              <td>OFFICER TRAINING - NSM Tactical Employment Course</td>
              <td>16 EA</td>
              <td>$720,000</td>
              <td>
                <div class="sc-dates">
                  <div>S</div>
                  <div class="date-range">SEP2025 - MAR2026</div>
                  <div>TA3</div>
                </div>
              </td>
              <td>A</td>
              <td>4</td>
            </tr>
            <tr>
              <td>006</td>
              <td>ENLISTED TRAINING - NSM Maintenance Course</td>
              <td>24 EA</td>
              <td>$672,000</td>
              <td>
                <div class="sc-dates">
                  <div>S</div>
                  <div class="date-range">OCT2025 - APR2026</div>
                  <div>TA3</div>
                </div>
              </td>
              <td>A</td>
              <td>4</td>
            </tr>
            <tr>
              <td>007</td>
              <td>TECHNICAL SERVICES - NSM Integration Support</td>
              <td>-</td>
              <td>$42,500,000</td>
              <td>
                <div class="sc-dates">
                  <div>S</div>
                  <div class="date-range">MAY2025 - DEC2027</div>
                  <div>TA3</div>
                </div>
              </td>
              <td>A</td>
              <td>4</td>
            </tr>
            <tr>
              <td>008</td>
              <td>LOGISTICS SUPPORT SERVICES</td>
              <td>-</td>
              <td>$18,400,000</td>
              <td>
                <div class="sc-dates">
                  <div>S</div>
                  <div class="date-range">JUN2025 - MAY2028</div>
                  <div>TA3</div>
                </div>
              </td>
              <td>A</td>
              <td>4</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue';
import TermTooltip from '@/components/TermTooltip.vue';

// Props will be defined here later, e.g., to receive case data
// defineProps({ caseData: Object });

// Date calculation helper (example for future dynamic implementation)
// const calculateDateRange = (implementationDate, startMonths, endMonths) => {
//   const startDate = new Date(implementationDate);
//   startDate.setMonth(startDate.getMonth() + startMonths);
//
//   const endDate = new Date(implementationDate);
//   endDate.setMonth(endDate.getMonth() + endMonths);
//
//   const formatDate = (date) => {
//     const month = String(date.getMonth() + 1).padStart(2, '0');
//     const day = String(date.getDate()).padStart(2, '0');
//     const year = date.getFullYear();
//     return `${month}/${day}/${year}`;
//   };
//
//   return `${formatDate(startDate)} - ${formatDate(endDate)}`;
// };
</script>

<style scoped>
.tab-content-section-wrapper {
  padding: 0;
  flex-grow: 1;
  display: flex;
  flex-direction: column;
  gap: 24px;
}

/* ===== ALERTS SECTION ===== */
.alerts-section {
  background: white;
  border: 2px solid #dc2626;
  border-radius: 16px;
  overflow: hidden;
  box-shadow: 0 2px 8px rgba(220, 38, 38, 0.1);
}

.alerts-header {
  background: white;
  padding: 12px 20px;
  font-weight: 700;
  color: #dc2626;
  font-size: 14px;
  border-bottom: 2px solid #dc2626;
}

.alerts-list {
  padding: 16px 20px;
  display: flex;
  flex-direction: column;
  gap: 12px;
  background: white;
}

.alert-item {
  display: flex;
  gap: 12px;
  align-items: flex-start;
  line-height: 1.6;
}

.alert-bullet {
  color: #dc2626;
  font-size: 12px;
  flex-shrink: 0;
  margin-top: 2px;
}

.alert-content {
  color: #1a1a1a;
  font-size: 13px;
}

.alert-content strong {
  font-weight: 600;
  color: #000;
}

.alert-link {
  color: #0066cc;
  text-decoration: none;
  font-weight: 500;
}

.alert-link:hover {
  text-decoration: underline;
}

/* FIXED: YELLOW highlight styling */
.yellow-highlight {
  background-color: #fef08a;
  color: #854d0e;
  padding: 2px 6px;
  border-radius: 3px;
  font-weight: 700;
}

/* ===== LINE ITEMS TABLE ===== */
.line-items-table-section {
  width: 100%;
  background: white;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

/* Single scroll container for both header and body */
.table-scroll-wrapper {
  width: 100%;
  max-height: 240px;
  overflow-y: auto;
  overflow-x: hidden;
  position: relative;
}

/* Custom scrollbar styling */
.table-scroll-wrapper::-webkit-scrollbar {
  width: 10px;
  height: 10px;
}

.table-scroll-wrapper::-webkit-scrollbar-track {
  background: #f1f1f1;
  border-radius: 5px;
}

.table-scroll-wrapper::-webkit-scrollbar-thumb {
  background: #888;
  border-radius: 5px;
}

.table-scroll-wrapper::-webkit-scrollbar-thumb:hover {
  background: #555;
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
  white-space: normal;
  word-wrap: break-word;
  position: relative;
}

.line-items-table th:last-child {
  border-right: none;
}

/* FIXED: Updated column width distribution - narrower MASL, wider Total Cost */
.col-line-nbr {
  width: 6%;
}

.col-masl {
  width: 18%; /* Narrowed from 22% */
}

.col-qty {
  width: 8%;
}

.col-total-cost {
  width: 15%; /* Increased from 13% to ensure full visibility */
}

.col-sc-mos-ta {
  width: 35%; /* Increased from 32% to accommodate new date format */
}

.col-orc {
  width: 7%;
}

.col-dtc {
  width: 11%;
}

.header-with-tooltip {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.subheader {
  font-size: 11px;
  font-weight: 500;
  color: #ffd700;
  font-style: italic;
  line-height: 1.2;
  white-space: normal;
  max-width: 200px;
}

.line-items-table tbody tr {
  border-bottom: 1px solid #e5e7eb;
}

.line-items-table tbody tr:nth-child(odd) {
  background: #f9fafb;
}

.line-items-table tbody tr:nth-child(even) {
  background: #ffffff;
}

.line-items-table tbody tr:hover {
  background: #f3f4f6;
}

.line-items-table td {
  padding: 12px 16px;
  border-right: 1px solid #e5e7eb;
  color: #1a1a1a;
  white-space: normal;
  word-wrap: break-word;
  vertical-align: top;
}

.line-items-table td:last-child {
  border-right: none;
}

/* SC/MOS/TA Date formatting */
.sc-dates {
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 12px;
  line-height: 1.4;
}

.sc-dates > div:first-child {
  font-weight: 600;
  color: #1a1a1a;
}

.date-range {
  color: #0066cc;
  font-weight: 500;
  font-size: 11px;
}

.sc-dates > div:last-child {
  font-weight: 600;
  color: #6b7280;
}

/* ===== CRITICAL: FORCE TOOLTIPS TO FLOAT ABOVE EVERYTHING ===== */

/* This is the nuclear option - tooltips MUST use position: fixed */
:deep(.v-popper__popper) {
  position: fixed !important;
  z-index: 999999 !important;
  /* Remove all transforms that could offset position */
  transform: none !important;
  /* Ensure it's positioned relative to viewport, not parent */
  inset: unset !important;
  left: unset !important;
  right: unset !important;
  top: unset !important;
  bottom: unset !important;
}

/* Force popper to recalculate using fixed strategy */
:deep(.v-popper__popper[data-popper-reference-hidden="false"]) {
  visibility: visible !important;
  opacity: 1 !important;
}

/* Ensure wrapper doesn't interfere */
:deep(.v-popper__wrapper) {
  display: inline-block;
}

/* Style the tooltip content */
:deep(.v-popper__inner) {
  background: #1f2937 !important;
  color: white !important;
  padding: 8px 12px !important;
  border-radius: 6px !important;
  font-size: 13px !important;
  line-height: 1.4 !important;
  max-width: 300px !important;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
  font-weight: 400 !important;
}

/* Arrow for top placement */
:deep(.v-popper__arrow-container) {
  position: absolute !important;
}

:deep(.v-popper__arrow-outer) {
  border-color: #1f2937 transparent transparent transparent !important;
}

:deep(.v-popper__arrow-inner) {
  border-color: #1f2937 transparent transparent transparent !important;
}

/* Alternative: Tippy.js support */
:deep(.tippy-box) {
  position: fixed !important;
  z-index: 999999 !important;
  background-color: #1f2937 !important;
  color: white !important;
  padding: 8px 12px !important;
  border-radius: 6px !important;
  font-size: 13px !important;
}

:deep(.tippy-arrow) {
  color: #1f2937 !important;
}

/* Generic tooltip support */
:deep([role="tooltip"]) {
  position: fixed !important;
  z-index: 999999 !important;
  background: #1f2937 !important;
  color: white !important;
  padding: 8px 12px !important;
  border-radius: 6px !important;
  font-size: 13px !important;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
}

/* Popper.js v2/v3 specific fixes */
:deep([data-popper-placement]) {
  position: fixed !important;
  z-index: 999999 !important;
}

:deep([data-popper-escaped]) {
  display: none !important;
}

/* Prevent overflow hidden from hiding tooltips */
.line-items-table-section,
.table-scroll-wrapper,
.line-items-table,
.line-items-table thead,
.line-items-table tbody {
  overflow: visible !important;
}

/* Re-apply overflow only where needed */
.table-scroll-wrapper {
  overflow: auto !important; /* Override the visible for scrolling */
}
</style>
