<script>
import { useCaseStore } from '@/stores/caseStore';
import { computed, watch, onMounted, ref } from 'vue';

export default {
  name: 'FinancialContent',
  
  props: {
    caseData: {
      type: Object,
      default: null
    }
  },
  
  setup(props) {
    const caseStore = useCaseStore();
    const isLoading = ref(false);
    const error = ref(null);
    const expandedRows = ref(new Set());
    const activeTooltip = ref(null);
    
    // Column tooltips with human-friendly explanations
    const columnTooltips = {
      'LINE': {
        title: 'Line Number',
        description: 'Row number in the financial table for easy reference.'
      },
      'ADJUSTED_NET_RSN': {
        title: 'Adjusted Net RSN',
        description: 'The total funding amount allocated to this RSN (Requisition Serial Number) after any adjustments.\n\nThis comes from the OA Received Amount column in the financial system.\n\nThink of it as: "How much money was approved for this line item?"'
      },
      'RSN': {
        title: 'RSN (Requisition Serial Number)',
        description: 'A unique identifier for each funding line in the case.\n\nEach RSN can have multiple PDLIs (items) underneath it.\n\nThink of it as: "The budget bucket that holds related items together."'
      },
      'PDLI_NUMBER': {
        title: 'PDLI Number',
        description: 'Price and Delivery Line Item identifier.\n\nThis is the specific item or service code within an RSN.\n\nThink of it as: "The individual item number within a budget bucket."'
      },
      'PDLI_DESCRIPTION': {
        title: 'PDLI Description',
        description: 'A brief description of what this line item covers.\n\nExamples: "NSM Tactical", "Procurement Support", "Training Services"'
      },
      'PDLI_DIRECTED_AMOUNT': {
        title: 'PDLI Directed Amount',
        description: 'The specific dollar amount allocated to this individual line item.\n\nThis comes from the Directed Reserve Amount in the financial system.\n\nThink of it as: "How much money is set aside for this specific item?"'
      },
      'OBLIGATED': {
        title: 'Obligated',
        description: 'Money that has been legally committed through contracts or purchase orders.\n\nOnce obligated, funds are legally bound to be spent.\n\nThink of it as: "Money we\'ve promised to pay for work or goods."'
      },
      'COMMITTED': {
        title: 'Committed',
        description: 'Money that has been earmarked or reserved but not yet legally obligated.\n\nThis is an internal hold on funds before contracts are signed.\n\nThink of it as: "Money we\'re planning to spend but haven\'t contracted yet."'
      },
      'PDLI_AVAILABLE_BALANCE': {
        title: 'PDLI Available Balance',
        description: 'How much money is still available for this specific line item.\n\nThink of it as: "How much is left to spend on this item?"',
        formula: 'PDLI Directed Amount − Obligated'
      },
      'TOTAL_AVAILABLE': {
        title: 'Total Available',
        description: 'The overall remaining balance for the entire RSN.\n\nThink of it as: "How much is left in this entire budget bucket?"',
        formula: 'Adjusted Net RSN − Obligated'
      }
    };
    
    const showTooltip = (columnKey) => {
      activeTooltip.value = columnKey;
    };
    
    const hideTooltip = () => {
      activeTooltip.value = null;
    };
    
    const getTooltip = (columnKey) => {
      return columnTooltips[columnKey] || null;
    };
    
    // Use store's active case
    const activeCase = computed(() => caseStore.activeCaseDetails);
    
    const currentCaseNumber = computed(() => {
      return activeCase.value?.caseNumber || activeCase.value?.id || null;
    });
    
    const financialDocuments = computed(() => {
      if (!activeCase.value) return [];
      
      const documents = 
        activeCase.value.caseDocuments || 
        activeCase.value.documents || 
        [];
      
      if (!Array.isArray(documents)) return [];
      
      return documents.filter(doc => 
        doc.documentType === 'FINANCIAL_DATA' ||
        doc.type === 'FINANCIAL_DATA' ||
        doc.metadata?.hasFinancialData === true ||
        (doc.metadata?.financialRecords && doc.metadata.financialRecords.length > 0)
      );
    });
    
    const hasFinancialData = computed(() => financialDocuments.value.length > 0);
    
    // =========================================================================
    // COLUMN MAPPING (per spec - keeping table column names the same):
    // =========================================================================
    // TABLE COLUMNS:
    //   - LINE #: Row index
    //   - ADJUSTED NET RSN: Column I from "2. MISIL RSN" (OA REC AMT) - per RSN
    //   - RSN: RSN identifier
    //   - PDLI NUMBER: PDLI identifier from Sheet 3
    //   - PDLI DESCRIPTION: PDLI DESC from Sheet 3
    //   - PDLI DIRECTED AMOUNT: Column H from "3. MISIL PDLI" (DIR RSRV AMT)
    //   - OBLIGATED: Column K from "2. MISIL RSN" (GROSS OBL AMT) - per RSN
    //   - COMMITTED: Column J from "2. MISIL RSN" (NET COMMIT AMT) - per RSN
    //   - PDLI AVAILABLE BALANCE: PDLI Directed - Obligated
    //   - TOTAL AVAILABLE: Adj Net RSN - Obligated
    // =========================================================================
    
    const groupedFinancialRecords = computed(() => {
      const records = [];
      
      financialDocuments.value.forEach(doc => {
        if (doc.metadata?.financialRecords) {
          records.push(...doc.metadata.financialRecords);
        }
      });
      
      if (records.length === 0) return [];
      
      // DEDUPLICATE records by RSN + PDLI combination
      const seen = new Set();
      const uniqueRecords = records.filter(record => {
        const rsn = record.rsn_identifier || record.rsn || '';
        const pdli = record.pdli_pdli || record.pdli || record.pdli_number || '';
        const key = `${rsn}-${pdli}`;
        if (seen.has(key)) {
          return false; // Skip duplicate
        }
        seen.add(key);
        return true;
      });
      
      console.log(`[FinancialContent] Total records: ${records.length}, Unique: ${uniqueRecords.length}`);
      
      const groups = {};
      uniqueRecords.forEach(record => {
        const rsn = record.rsn_identifier || record.rsn || record.line_item || 'N/A';
        
        if (!groups[rsn]) {
          groups[rsn] = {
            rsn: rsn,
            records: [],
            // RSN-level totals (from Sheet 2 - MISIL RSN)
            totalAdjNetRsn: 0,       // OA REC AMT (Column I)
            totalCommitted: 0,        // NET COMMIT AMT (Column J)
            totalObligated: 0,        // GROSS OBL AMT (Column K)
            totalExpended: 0,         // NET EXP AMT (Column L)
            // PDLI-level totals (from Sheet 3 - MISIL PDLI)
            totalPdliDirected: 0      // DIR RSRV AMT (Column H)
          };
        }
        
        const enrichedRecord = {
          ...record,
          pdli_number: record.pdli_pdli || record.pdli || record.PDLI_pdli || 'N/A'
        };
        
        groups[rsn].records.push(enrichedRecord);
        
        // =================================================================
        // DATA SOURCE MAPPING (per spec):
        // =================================================================
        
        // ADJUSTED NET RSN: Column I from "2. MISIL RSN" (OA REC AMT)
        // This is stored per RSN, so we use the first record's value (same for all PDLIs in RSN)
        const adjNetRsn = parseFloat(record.adjusted_net_rsn || record.oa_rec_amt || 0);
        if (groups[rsn].records.length === 1) {
          // Only set once per RSN (all PDLIs under same RSN have same RSN-level values)
          groups[rsn].totalAdjNetRsn = adjNetRsn;
        }
        
        // COMMITTED: Column J from "2. MISIL RSN" (NET COMMIT AMT)
        const committed = parseFloat(record.committed || record.net_commit_amt || 0);
        if (groups[rsn].records.length === 1) {
          groups[rsn].totalCommitted = committed;
        }
        
        // OBLIGATED (RSN-level): Column K from "2. MISIL RSN" (GROSS OBL AMT)
        const obligated = parseFloat(record.obligated || 0);
        if (groups[rsn].records.length === 1) {
          groups[rsn].totalObligated = obligated;
        }
        
        // EXPENDED: Column L from "2. MISIL RSN" (NET EXP AMT)
        const expended = parseFloat(record.expended || record.net_exp_amt || 0);
        if (groups[rsn].records.length === 1) {
          groups[rsn].totalExpended = expended;
        }
        
        // PDLI DIRECTED AMOUNT: Column H from "3. MISIL PDLI" (DIR RSRV AMT)
        // This is summed across all PDLIs in the RSN
        groups[rsn].totalPdliDirected += parseFloat(record.pdli_directed_amt || record.dir_rsrv_amt || 0);
        
        // PDLI OBLIGATED: Column I from "3. MISIL PDLI" (NET OBL AMT)
        // Sum of PDLI-level obligated amounts for PDLI Available calculation
        if (!groups[rsn].totalPdliObligated) groups[rsn].totalPdliObligated = 0;
        groups[rsn].totalPdliObligated += parseFloat(record.pdli_obligated || record.net_obl_amt || 0);
      });
      
      // Calculate derived fields for each group
      Object.values(groups).forEach(group => {
        // Sum up PDLI-level values for RSN row display
        let sumPdliCommitted = 0;
        let sumPdliObligated = 0;
        let sumPdliExpended = 0;
        let sumPdliAvailable = 0;
        
        group.records.forEach(record => {
          const pdliDirected = parseFloat(record.pdli_directed_amt || record.dir_rsrv_amt || 0);
          const pdliCommitted = parseFloat(record.pdli_net_commit_amt || 0);
          const pdliNetObl = parseFloat(record.pdli_obligated || record.net_obl_amt || 0);
          const pdliNetExp = parseFloat(record.pdli_expended || record.net_exp_amt || 0);
          
          sumPdliCommitted += pdliCommitted;
          sumPdliObligated += pdliNetObl;
          sumPdliExpended += pdliNetExp;
          
          // PDLI Available = Directed - (Committed + Obligated + Expended)
          const pdliAvail = parseFloat(record.pdli_available_bal || (pdliDirected - (pdliCommitted + pdliNetObl + pdliNetExp)));
          sumPdliAvailable += pdliAvail;
        });
        
        // RSN-level PDLI Available Balance = Sum of all PDLI available balances
        group.pdliAvailable = sumPdliAvailable;
        
        // TOTAL AVAILABLE = Adj Net RSN - (Obligated + Committed) at RSN level
        group.totalAvailable = group.totalAdjNetRsn - (group.totalObligated + group.totalCommitted);
      });
      
      return Object.values(groups).sort((a, b) => {
        const aNum = parseInt(a.rsn) || 0;
        const bNum = parseInt(b.rsn) || 0;
        return aNum - bNum;
      });
    });
    
    const totalRecordCount = computed(() => {
      return groupedFinancialRecords.value.reduce((sum, group) => sum + group.records.length, 0);
    });
    
    // Calculate balances for individual PDLI records (detail rows)
    const calculateRecordBalances = (record) => {
      // PDLI Directed = Column H from "3. MISIL PDLI" (DIR RSRV AMT)
      const pdliDirected = parseFloat(record.pdli_directed_amt || record.dir_rsrv_amt || 0);
      // Net Committed = Column I from "3. MISIL PDLI" (NET COMMIT AMT)
      const pdliCommitted = parseFloat(record.pdli_net_commit_amt || 0);
      // Net Obligated = Column J from "3. MISIL PDLI" (NET OBL AMT)
      const pdliNetObl = parseFloat(record.pdli_obligated || record.net_obl_amt || 0);
      // Net Expended = Column K from "3. MISIL PDLI" (NET EXP AMT)
      const pdliNetExp = parseFloat(record.pdli_expended || record.net_exp_amt || 0);
      
      // Gross Obligated = NET OBL AMT + NET EXP AMT (Columns J + K)
      const grossObligated = parseFloat(record.pdli_gross_obligated || (pdliNetObl + pdliNetExp));
      
      // PDLI Available = Directed - (Committed + Net Obl + Net Exp)
      // EX Line 005: 720,000 - (180,000 + 180,000 + 180,000) = 180,000
      const pdliAvailable = parseFloat(record.pdli_available_bal || (pdliDirected - (pdliCommitted + pdliNetObl + pdliNetExp)));
      
      return {
        pdliCommitted: pdliCommitted,
        grossObligated: grossObligated,
        pdliAvailable: pdliAvailable,
        // Total Available at PDLI level should be 0
        totalAvailable: 0
      };
    };
    
    const toggleRow = (rsn) => {
      if (expandedRows.value.has(rsn)) {
        expandedRows.value.delete(rsn);
      } else {
        expandedRows.value.add(rsn);
      }
      expandedRows.value = new Set(expandedRows.value);
    };
    
    const isRowExpanded = (rsn) => {
      return expandedRows.value.has(rsn);
    };
    
    const refreshData = async () => {
      const caseId = currentCaseNumber.value;
      if (!caseId) return;
      
      console.log(`[FinancialContent] Refreshing data for: ${caseId}`);
      isLoading.value = true;
      error.value = null;
      
      try {
        await caseStore.fetchCaseDetails(caseId);
      } catch (err) {
        console.error('[FinancialContent] Error:', err);
        error.value = err.message;
      } finally {
        isLoading.value = false;
      }
    };
    
    const formatCurrency = (value) => {
      if (value == null || value === 0) return '–';
      return '$' + Number(value).toLocaleString('en-US', {
        minimumFractionDigits: 0,
        maximumFractionDigits: 0
      });
    };
    
    const getAvailableClass = (value) => {
      if (value == null) return '';
      if (value > 0) return 'positive';
      if (value < 0) return 'negative';
      return '';
    };
    
    watch(() => caseStore.activeCaseId, (newId) => {
      if (newId) refreshData();
    });
    
    onMounted(() => {
      if (caseStore.activeCaseId) refreshData();
    });
    
    return {
      isLoading,
      error,
      currentCaseNumber,
      hasFinancialData,
      groupedFinancialRecords,
      totalRecordCount,
      expandedRows,
      toggleRow,
      isRowExpanded,
      refreshData,
      formatCurrency,
      getAvailableClass,
      calculateRecordBalances,
      // Tooltip functions
      activeTooltip,
      showTooltip,
      hideTooltip,
      getTooltip,
      columnTooltips
    };
  }
};
</script>

<template>
  <div class="financial-content">
    <!-- Loading State -->
    <div v-if="isLoading" class="loading-state">
      <div class="spinner"></div>
      <p>Loading financial data...</p>
    </div>
    
    <!-- Error State -->
    <div v-else-if="error" class="error-state">
      <i class="fas fa-exclamation-triangle"></i>
      <p>Error loading financial data</p>
      <small>{{ error }}</small>
      <button class="retry-btn" @click="refreshData">
        <i class="fas fa-redo"></i> Retry
      </button>
    </div>
    
    <!-- No Data State -->
    <div v-else-if="!hasFinancialData" class="no-data-state">
      <i class="fas fa-file-invoice-dollar"></i>
      <h3>No Financial Data Available</h3>
      <p>Upload a MISIL Excel file to see financial records</p>
      <p class="hint">Go to the Documents tab to upload financial documents</p>
    </div>
    
    <!-- Financial Data Display -->
    <div v-else class="financial-data-display">
      <div class="data-header">
        <h3>Financial Data - {{ currentCaseNumber }}</h3>
        <span class="record-badge">
          {{ totalRecordCount }} records across {{ groupedFinancialRecords.length }} RSNs
        </span>
      </div>
      
      <div class="table-container">
        <table class="financial-table">
          <thead>
            <tr>
              <th class="expand-col"></th>
              
              <!-- LINE # -->
              <th class="line-col header-with-tooltip">
                <span class="header-text">LINE #</span>
                <i class="fas fa-info-circle tooltip-icon"></i>
                <div class="column-tooltip">
                  <div class="tooltip-title">Line Number</div>
                  <div class="tooltip-desc">Row number in the financial table for easy reference.</div>
                </div>
              </th>
              
              <!-- ADJUSTED NET RSN -->
              <th class="amount-col header-with-tooltip">
                <span class="header-text">ADJUSTED<br>NET RSN</span>
                <i class="fas fa-info-circle tooltip-icon"></i>
                <div class="column-tooltip">
                  <div class="tooltip-title">Adjusted Net RSN</div>
                  <div class="tooltip-desc">The total funding amount allocated to this RSN after adjustments.

Think of it as: "How much money was approved for this line item?"</div>
                </div>
              </th>
              
              <!-- RSN -->
              <th class="rsn-col header-with-tooltip">
                <span class="header-text">RSN</span>
                <i class="fas fa-info-circle tooltip-icon"></i>
                <div class="column-tooltip">
                  <div class="tooltip-title">RSN (Requisition Serial Number)</div>
                  <div class="tooltip-desc">A unique identifier for each funding line in the case.

Each RSN can have multiple PDLIs (items) underneath it.

Think of it as: "The budget bucket that holds related items together."</div>
                </div>
              </th>
              
              <!-- PDLI NUMBER -->
              <th class="pdli-num-col header-with-tooltip">
                <span class="header-text">PDLI<br>NUMBER</span>
                <i class="fas fa-info-circle tooltip-icon"></i>
                <div class="column-tooltip">
                  <div class="tooltip-title">PDLI Number</div>
                  <div class="tooltip-desc">Price and Delivery Line Item identifier.

This is the specific item or service code within an RSN.

Think of it as: "The individual item number within a budget bucket."</div>
                </div>
              </th>
              
              <!-- PDLI DESCRIPTION -->
              <th class="desc-col header-with-tooltip">
                <span class="header-text">PDLI<br>DESCRIPTION</span>
                <i class="fas fa-info-circle tooltip-icon"></i>
                <div class="column-tooltip">
                  <div class="tooltip-title">PDLI Description</div>
                  <div class="tooltip-desc">A brief description of what this line item covers.

Examples: "NSM Tactical", "Procurement Support", "Training Services"</div>
                </div>
              </th>
              
              <!-- PDLI DIRECTED AMOUNT -->
              <th class="amount-col header-with-tooltip">
                <span class="header-text">PDLI DIRECTED<br>AMOUNT</span>
                <i class="fas fa-info-circle tooltip-icon"></i>
                <div class="column-tooltip">
                  <div class="tooltip-title">PDLI Directed Amount</div>
                  <div class="tooltip-desc">The specific dollar amount allocated to this individual line item.

Think of it as: "How much money is set aside for this specific item?"</div>
                </div>
              </th>
              
              <!-- COMMITTED -->
              <th class="amount-col header-with-tooltip">
                <span class="header-text">COMMITTED</span>
                <i class="fas fa-info-circle tooltip-icon"></i>
                <div class="column-tooltip">
                  <div class="tooltip-title">Committed</div>
                  <div class="tooltip-desc">Money that has been earmarked or reserved but not yet legally obligated.

This is an internal hold on funds before contracts are signed.

Think of it as: "Money we're planning to spend but haven't contracted yet."</div>
                </div>
              </th>
              
              <!-- OBLIGATED -->
              <th class="amount-col header-with-tooltip">
                <span class="header-text">OBLIGATED</span>
                <i class="fas fa-info-circle tooltip-icon"></i>
                <div class="column-tooltip">
                  <div class="tooltip-title">Obligated</div>
                  <div class="tooltip-desc">Money that has been legally committed through contracts or purchase orders.

Once obligated, funds are legally bound to be spent.

Think of it as: "Money we've promised to pay for work or goods."</div>
                </div>
              </th>
              
              <!-- PDLI AVAILABLE BALANCE -->
              <th class="amount-col header-with-tooltip">
                <span class="header-text">PDLI AVAILABLE<br>BALANCE</span>
                <i class="fas fa-info-circle tooltip-icon"></i>
                <div class="column-tooltip tooltip-left">
                  <div class="tooltip-title">PDLI Available Balance</div>
                  <div class="tooltip-desc">How much money is still available for this specific line item.

Think of it as: "How much is left to spend on this item?"</div>
                  <div class="tooltip-formula">
                    <strong>📊 Formula:</strong> PDLI Directed Amount − Obligated
                  </div>
                </div>
              </th>
              
              <!-- TOTAL AVAILABLE -->
              <th class="amount-col header-with-tooltip">
                <span class="header-text">TOTAL<br>AVAILABLE</span>
                <i class="fas fa-info-circle tooltip-icon"></i>
                <div class="column-tooltip tooltip-left">
                  <div class="tooltip-title">Total Available</div>
                  <div class="tooltip-desc">The overall remaining balance for the entire RSN.

Think of it as: "How much is left in this entire budget bucket?"</div>
                  <div class="tooltip-formula">
                    <strong>📊 Formula:</strong> Adjusted Net RSN − Obligated
                  </div>
                </div>
              </th>
            </tr>
          </thead>
          <tbody>
            <template v-for="(group, groupIndex) in groupedFinancialRecords" :key="group.rsn">
              <!-- RSN Summary Row (Group Header) -->
              <tr class="group-row" @click="toggleRow(group.rsn)">
                <td class="expand-col">
                  <button class="expand-btn" @click.stop="toggleRow(group.rsn)">
                    <i :class="isRowExpanded(group.rsn) ? 'fas fa-caret-down' : 'fas fa-caret-right'"></i>
                  </button>
                </td>
                <td class="line-col">{{ groupIndex + 1 }}</td>
                <!-- ADJUSTED NET RSN: OA REC AMT from Sheet 2 (per RSN) -->
                <td class="amount-col amount primary">{{ formatCurrency(group.totalAdjNetRsn) }}</td>
                <td class="rsn-col">
                  <strong>RSN {{ String(group.rsn).padStart(3, '0') }}</strong>
                  <span class="pdli-count">({{ group.records.length }} PDLIs)</span>
                </td>
                <td class="pdli-num-col">–</td>
                <td class="desc-col"></td>
                <!-- PDLI DIRECTED: Sum of DIR RSRV AMT from Sheet 3 -->
                <td class="amount-col amount">{{ formatCurrency(group.totalPdliDirected) }}</td>
                <!-- COMMITTED: NET COMMIT AMT from Sheet 2 (per RSN) -->
                <td class="amount-col amount">{{ formatCurrency(group.totalCommitted) }}</td>
                <!-- OBLIGATED: GROSS OBL AMT from Sheet 2 (per RSN) -->
                <td class="amount-col amount">{{ formatCurrency(group.totalObligated) }}</td>
                <!-- PDLI AVAILABLE: PDLI Directed - Obligated -->
                <td class="amount-col amount" :class="getAvailableClass(group.pdliAvailable)">
                  {{ formatCurrency(group.pdliAvailable) }}
                </td>
                <!-- TOTAL AVAILABLE: Adj Net RSN - Obligated -->
                <td class="amount-col amount" :class="getAvailableClass(group.totalAvailable)">
                  {{ formatCurrency(group.totalAvailable) }}
                </td>
              </tr>
              
              <!-- PDLI Detail Rows (Expanded) -->
              <template v-if="isRowExpanded(group.rsn)">
                <tr 
                  v-for="(record, index) in group.records" 
                  :key="`${group.rsn}-${index}`"
                  class="detail-row"
                >
                  <td class="expand-col"></td>
                  <td class="line-col">–</td>
                  <!-- Adj Net RSN only shown at RSN level -->
                  <td class="amount-col amount">–</td>
                  <td class="rsn-col detail-rsn">{{ record.rsn_identifier || record.rsn || group.rsn }}</td>
                  <td class="pdli-num-col">{{ record.pdli_number || record.pdli || '–' }}</td>
                  <td class="desc-col">{{ record.pdli_description || record.pdli_desc || 'N/A' }}</td>
                  <!-- PDLI Directed for this specific PDLI -->
                  <td class="amount-col amount">{{ formatCurrency(record.pdli_directed_amt || record.dir_rsrv_amt) }}</td>
                  <!-- Net Committed from Sheet 3 -->
                  <td class="amount-col amount">{{ formatCurrency(calculateRecordBalances(record).pdliCommitted) }}</td>
                  <!-- Gross Obligated = NET OBL AMT + NET EXP AMT from Sheet 3 -->
                  <td class="amount-col amount">{{ formatCurrency(calculateRecordBalances(record).grossObligated) }}</td>
                  <!-- PDLI Available = Directed - (Committed + Obligated + Expended) -->
                  <td class="amount-col amount" :class="getAvailableClass(calculateRecordBalances(record).pdliAvailable)">
                    {{ formatCurrency(calculateRecordBalances(record).pdliAvailable) }}
                  </td>
                  <!-- Total Available at PDLI level = 0 -->
                  <td class="amount-col amount">{{ formatCurrency(0) }}</td>
                </tr>
              </template>
            </template>
          </tbody>
        </table>
      </div>
      
      <!-- Legend/Help Section -->
      <div class="formula-legend">
        <h4><i class="fas fa-calculator"></i> How Balances Are Calculated</h4>
        <div class="formula-grid">
          <div class="formula-item">
            <span class="formula-name">RSN Total Available</span>
            <span class="formula-equals">=</span>
            <span class="formula-calc">Adjusted Net RSN − (Obligated + Committed)</span>
          </div>
          <div class="formula-item">
            <span class="formula-name">PDLI Available Balance</span>
            <span class="formula-equals">=</span>
            <span class="formula-calc">PDLI Directed − (Committed + Obligated + Expended)</span>
          </div>
        </div>
        <div class="status-legend">
          <h5>Case Status Colors</h5>
          <div class="status-items">
            <span class="status-item"><span class="status-dot open"></span> Open (Active)</span>
            <span class="status-item"><span class="status-dot ssc"></span> Supply/Services Complete</span>
            <span class="status-item"><span class="status-dot closed"></span> Interim Closed</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.financial-content {
  padding: 20px;
  height: 100%;
  overflow-y: auto;
  background: #f8f9fa;
}

.loading-state,
.error-state,
.no-data-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 400px;
  text-align: center;
  color: #666;
}

.loading-state .spinner {
  width: 40px;
  height: 40px;
  border: 4px solid #f3f3f3;
  border-top: 4px solid #3498db;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin-bottom: 20px;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.error-state { color: #e74c3c; }
.error-state i { font-size: 48px; margin-bottom: 20px; }
.no-data-state i { font-size: 64px; color: #bdc3c7; margin-bottom: 20px; }
.no-data-state h3 { margin: 10px 0; color: #2c3e50; }
.no-data-state .hint { color: #95a5a6; font-size: 14px; margin-top: 10px; }

.retry-btn {
  margin-top: 20px;
  padding: 10px 20px;
  background: #3498db;
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 14px;
  font-weight: 500;
  display: inline-flex;
  align-items: center;
  gap: 8px;
}
.retry-btn:hover { background: #2980b9; }

.financial-data-display { animation: fadeIn 0.3s ease-in; }
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }

.data-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.data-header h3 {
  margin: 0;
  color: #2c3e50;
  font-size: 1.1rem;
  font-weight: 600;
}

.record-badge {
  background: #3498db;
  color: white;
  padding: 6px 14px;
  border-radius: 20px;
  font-size: 0.8rem;
  font-weight: 500;
}

.table-container {
  overflow-x: auto;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
  background: white;
}

.financial-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.8rem;
  min-width: 1100px;
}

.financial-table thead {
  background: #2c3e50;
  color: white;
}

.financial-table th {
  padding: 12px 8px;
  text-align: left;
  font-weight: 600;
  font-size: 0.65rem;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  position: relative;
  vertical-align: top;
}

.financial-table td {
  padding: 10px 8px;
  border-bottom: 1px solid #e8ecef;
}

/* Header with tooltip styling */
.header-with-tooltip {
  cursor: help;
  position: relative;
}

.header-text {
  display: inline-block;
  line-height: 1.3;
}

.tooltip-icon {
  font-size: 0.55rem;
  opacity: 0.7;
  margin-left: 3px;
  vertical-align: top;
  color: #8ecaff;
}

.header-with-tooltip:hover .tooltip-icon {
  opacity: 1;
  color: #fff;
}

/* Column tooltip popup - CSS only approach */
.column-tooltip {
  display: none;
  position: absolute;
  top: 100%;
  left: 0;
  z-index: 9999;
  background: #ffffff;
  color: #333;
  border: 1px solid #ccc;
  border-radius: 8px;
  box-shadow: 0 6px 24px rgba(0,0,0,0.25);
  padding: 14px 16px;
  min-width: 260px;
  max-width: 300px;
  margin-top: 6px;
  text-align: left;
  text-transform: none;
  letter-spacing: normal;
  font-size: 0.82rem;
  font-weight: 400;
}

/* Show tooltip on hover */
.header-with-tooltip:hover .column-tooltip {
  display: block !important;
}

.column-tooltip.tooltip-left {
  left: auto;
  right: 0;
}

.column-tooltip::before {
  content: '';
  position: absolute;
  top: -8px;
  left: 20px;
  border-left: 8px solid transparent;
  border-right: 8px solid transparent;
  border-bottom: 8px solid #ffffff;
}

.column-tooltip.tooltip-left::before {
  left: auto;
  right: 20px;
}

.tooltip-title {
  font-weight: 700;
  font-size: 0.9rem;
  color: #1a5276;
  margin-bottom: 8px;
  padding-bottom: 6px;
  border-bottom: 2px solid #3498db;
}

.tooltip-desc {
  font-size: 0.8rem;
  line-height: 1.6;
  color: #444;
  white-space: pre-line;
}

.tooltip-formula {
  margin-top: 10px;
  padding: 10px;
  background: #e8f4fd;
  border-radius: 4px;
  font-size: 0.78rem;
  color: #2980b9;
  border-left: 3px solid #3498db;
}

.tooltip-formula strong {
  color: #1a5276;
}

.expand-col { width: 40px; text-align: center; }
.line-col { width: 50px; text-align: center; }
.rsn-col { width: 110px; }
.pdli-num-col { width: 80px; }
.desc-col { width: 140px; }
.amount-col { width: 100px; text-align: right; }

.group-row {
  background: #f8f9fa;
  cursor: pointer;
  transition: background-color 0.2s;
}
.group-row:hover { background: #e9ecef; }

.detail-row { background: white; }
.detail-row:hover { background: #f8f9fa; }

.expand-btn {
  background: transparent;
  border: none;
  width: 24px;
  height: 24px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  color: #666;
  font-size: 1rem;
}
.expand-btn:hover { color: #3498db; }

.rsn-col strong { color: #2c3e50; font-weight: 600; }
.pdli-count { margin-left: 6px; color: #888; font-size: 0.7rem; font-weight: normal; }
.detail-rsn { padding-left: 8px; color: #666; }

.amount {
  font-family: 'SF Mono', 'Consolas', monospace;
  font-weight: 500;
  text-align: right;
}

.amount.primary { color: #e67e22; font-weight: 600; }
.amount.positive { color: #27ae60; }
.amount.negative { color: #e74c3c; }

/* Formula Legend */
.formula-legend {
  margin-top: 20px;
  padding: 16px 20px;
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}

.formula-legend h4 {
  margin: 0 0 12px 0;
  font-size: 0.9rem;
  color: #2c3e50;
  display: flex;
  align-items: center;
  gap: 8px;
}

.formula-legend h4 i {
  color: #3498db;
}

.formula-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 20px;
}

.formula-item {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 0.8rem;
  padding: 8px 12px;
  background: #f8f9fa;
  border-radius: 6px;
}

.formula-name {
  font-weight: 600;
  color: #2c3e50;
}

.formula-equals {
  color: #888;
}

.formula-calc {
  color: #3498db;
  font-family: 'SF Mono', 'Consolas', monospace;
}

.status-legend {
  margin-top: 16px;
  padding-top: 12px;
  border-top: 1px solid #eee;
}

.status-legend h5 {
  margin: 0 0 10px 0;
  font-size: 0.8rem;
  color: #666;
  font-weight: 600;
}

.status-items {
  display: flex;
  gap: 20px;
  flex-wrap: wrap;
}

.status-item {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 0.75rem;
  color: #555;
}

.status-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
}

.status-dot.open {
  background: #27ae60;
}

.status-dot.ssc {
  background: #f39c12;
}

.status-dot.closed {
  background: #e74c3c;
}
</style>