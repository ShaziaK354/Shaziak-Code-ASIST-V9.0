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
    
    const groupedFinancialRecords = computed(() => {
      const records = [];
      
      financialDocuments.value.forEach(doc => {
        if (doc.metadata?.financialRecords) {
          records.push(...doc.metadata.financialRecords);
        }
      });
      
      if (records.length === 0) return [];
      
      const groups = {};
      records.forEach(record => {
        const rsn = record.rsn_identifier || record.rsn || record.line_item || 'N/A';
        
        if (!groups[rsn]) {
          groups[rsn] = {
            rsn: rsn,
            records: [],
            totalOaRec: 0,
            totalNetCommit: 0,
            totalGrossObl: 0,
            totalNetExp: 0,
            totalPdliDirected: 0
          };
        }
        
        const enrichedRecord = {
          ...record,
          pdli_number: record.pdli_pdli || record.pdli || record.PDLI_pdli || 'N/A'
        };
        
        groups[rsn].records.push(enrichedRecord);
        
        // =================================================================
        // DATA SOURCE MAPPING (per requirements):
        // =================================================================
        // ADJUSTED NET RSN (totalOaRec):
        //   Source: Column L on "1. Case, SSC Preclosure" sheet
        //   Field: adjusted_net_rsn (mapped from oa_rec_amt)
        // -----------------------------------------------------------------
        groups[rsn].totalOaRec += parseFloat(record.adjusted_net_rsn || record.oa_rec_amt || 0);
        
        // COMMITTED (totalNetCommit):
        //   Source: Column M on "1. Case, SSC Preclosure" sheet
        //   Field: committed (mapped from net_commit_amt)
        // -----------------------------------------------------------------
        groups[rsn].totalNetCommit += parseFloat(record.committed || record.net_commit_amt || 0);
        
        // OBLIGATED (totalGrossObl):
        //   Source: Column N on "1. Case, SSC Preclosure" sheet
        //   Field: obligated (mapped from net_obl_amt)
        // -----------------------------------------------------------------
        groups[rsn].totalGrossObl += parseFloat(record.obligated || record.net_obl_amt || 0);
        
        // EXPENDED (totalNetExp):
        //   Source: Column K on "2. MISIL RSN" sheet
        //   Field: net_exp_amt
        // -----------------------------------------------------------------
        groups[rsn].totalNetExp += parseFloat(record.net_exp_amt || 0);
        
        // PDLI DIRECTED AMOUNT (totalPdliDirected):
        //   Source: Column H on "3. MISIL PDLI" sheet
        //   Field: pdli_directed_amt (mapped from dir_rsrv_amt)
        // -----------------------------------------------------------------
        groups[rsn].totalPdliDirected += parseFloat(record.pdli_directed_amt || record.dir_rsrv_amt || 0);
      });
      
      Object.values(groups).forEach(group => {
        // PDLI AVAILABLE BALANCE = PDLI Directed - Committed - Obligated
        group.pdliAvailable = group.totalPdliDirected - group.totalNetCommit - group.totalGrossObl;
        
        // TOTAL AVAILABLE = Adj Net RSN Total - Committed
        // (per spec: Adj Net RSN Total - Column M)
        group.totalAvailable = group.totalOaRec - group.totalNetCommit;
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
    
    const calculateRecordBalances = (record) => {
      // Adj Net RSN = Column L from "1. Case, SSC Preclosure"
      const oaRec = parseFloat(record.adjusted_net_rsn || record.oa_rec_amt || 0);
      // Committed = Column M from "1. Case, SSC Preclosure"
      const netCommit = parseFloat(record.committed || record.net_commit_amt || 0);
      // Obligated = Column N from "1. Case, SSC Preclosure"
      const netObl = parseFloat(record.obligated || record.net_obl_amt || 0);
      // PDLI Directed = Column H from "3. MISIL PDLI"
      const pdliDirected = parseFloat(record.pdli_directed_amt || record.dir_rsrv_amt || 0);
      
      return {
        // PDLI Available = PDLI Directed - Committed - Obligated
        pdliAvailable: pdliDirected - netCommit - netObl,
        // Total Available = Adj Net RSN - Committed
        totalAvailable: oaRec - netCommit
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
      return Number(value) >= 0 ? 'positive' : 'negative';
    };
    
    // Watch for case changes in store
    watch(() => caseStore.activeCaseDetails?.caseNumber, (newId, oldId) => {
      if (newId && newId !== oldId) {
        console.log(`[FinancialContent] Case changed: ${oldId} -> ${newId}`);
        expandedRows.value = new Set();
      }
    });
    
    // Watch store loading state
    watch(() => caseStore.isLoadingCases, (loading) => {
      isLoading.value = loading;
    });
    
    onMounted(() => {
      console.log('[FinancialContent] Mounted, case:', currentCaseNumber.value);
      // Initial loading state based on store
      isLoading.value = caseStore.isLoadingCases;
    });
    
    return {
      activeCase,
      currentCaseNumber,
      financialDocuments,
      hasFinancialData,
      groupedFinancialRecords,
      totalRecordCount,
      isLoading,
      error,
      formatCurrency,
      getAvailableClass,
      toggleRow,
      isRowExpanded,
      calculateRecordBalances,
      refreshData,
      caseStore
    };
  }
};
</script>

<template>
  <div class="financial-content">
    <!-- Loading: check both local and store loading states -->
    <div v-if="isLoading || caseStore.isLoadingCases" class="loading-state">
      <div class="spinner"></div>
      <p>Loading financial data{{ currentCaseNumber ? ` for ${currentCaseNumber}` : '' }}...</p>
    </div>
    
    <!-- Error state -->
    <div v-else-if="error" class="error-state">
      <i class="fas fa-exclamation-circle"></i>
      <p>{{ error }}</p>
      <button @click="refreshData" class="retry-btn">
        <i class="fas fa-sync-alt"></i> Retry
      </button>
    </div>
    
    <!-- No case selected -->
    <div v-else-if="!currentCaseNumber" class="no-data-state">
      <i class="fas fa-folder-open"></i>
      <h3>No Case Selected</h3>
      <p>Select a case from the sidebar to view financial data.</p>
    </div>
    
    <!-- No financial data for this case -->
    <div v-else-if="!hasFinancialData" class="no-data-state">
      <i class="fas fa-chart-line"></i>
      <h3>No Financial Data</h3>
      <p>No financial data found for case {{ currentCaseNumber }}.</p>
      <p class="hint">Upload MISIL PDLI data to view financial records.</p>
      <button @click="refreshData" class="retry-btn">
        <i class="fas fa-sync-alt"></i> Refresh
      </button>
    </div>
    
    <!-- Show financial data -->
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
              <th class="line-col">LINE #</th>
              <th class="amount-col">ADJUSTED NET RSN</th>
              <th class="rsn-col">RSN</th>
              <th class="pdli-num-col">PDLI NUMBER</th>
              <th class="desc-col">PDLI DESCRIPTION</th>
              <th class="amount-col">PDLI DIRECTED AMOUNT</th>
              <th class="amount-col">OBLIGATED</th>
              <th class="amount-col">COMMITTED</th>
              <th class="amount-col">PDLI AVAILABLE BALANCE</th>
              <th class="amount-col">TOTAL AVAILABLE</th>
            </tr>
          </thead>
          <tbody>
            <template v-for="(group, groupIndex) in groupedFinancialRecords" :key="group.rsn">
              <tr class="group-row" @click="toggleRow(group.rsn)">
                <td class="expand-col">
                  <button class="expand-btn" @click.stop="toggleRow(group.rsn)">
                    <i :class="isRowExpanded(group.rsn) ? 'fas fa-caret-down' : 'fas fa-caret-right'"></i>
                  </button>
                </td>
                <td class="line-col">{{ groupIndex + 1 }}</td>
                <td class="amount-col amount primary">{{ formatCurrency(group.totalOaRec) }}</td>
                <td class="rsn-col">
                  <strong>RSN {{ String(group.rsn).padStart(3, '0') }}</strong>
                  <span class="pdli-count">({{ group.records.length }} PDLIs)</span>
                </td>
                <td class="pdli-num-col">–</td>
                <td class="desc-col">RSN Total</td>
                <td class="amount-col amount">{{ formatCurrency(group.totalPdliDirected) }}</td>
                <td class="amount-col amount">{{ formatCurrency(group.totalGrossObl) }}</td>
                <td class="amount-col amount">{{ formatCurrency(group.totalNetCommit) }}</td>
                <td class="amount-col amount" :class="getAvailableClass(group.pdliAvailable)">
                  {{ formatCurrency(group.pdliAvailable) }}
                </td>
                <td class="amount-col amount" :class="getAvailableClass(group.totalAvailable)">
                  {{ formatCurrency(group.totalAvailable) }}
                </td>
              </tr>
              
              <template v-if="isRowExpanded(group.rsn)">
                <tr 
                  v-for="(record, index) in group.records" 
                  :key="`${group.rsn}-${index}`"
                  class="detail-row"
                >
                  <td class="expand-col"></td>
                  <td class="line-col">–</td>
                  <td class="amount-col amount">–</td>
                  <td class="rsn-col detail-rsn">{{ record.rsn_identifier || record.rsn || group.rsn }}</td>
                  <td class="pdli-num-col">{{ record.pdli_number || record.pdli || '–' }}</td>
                  <td class="desc-col">{{ record.pdli_description || record.pdli_desc || 'N/A' }}</td>
                  <td class="amount-col amount">{{ formatCurrency(record.pdli_directed_amt || record.dir_rsrv_amt) }}</td>
                  <td class="amount-col amount">{{ formatCurrency(record.obligated || record.net_obl_amt) }}</td>
                  <td class="amount-col amount">{{ formatCurrency(record.committed || record.net_commit_amt) }}</td>
                  <td class="amount-col amount" :class="getAvailableClass(calculateRecordBalances(record).pdliAvailable)">
                    {{ formatCurrency(calculateRecordBalances(record).pdliAvailable) }}
                  </td>
                  <td class="amount-col amount">–</td>
                </tr>
              </template>
            </template>
          </tbody>
        </table>
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
  min-width: 1200px;
}

.financial-table thead {
  background: #2c3e50;
  color: white;
}

.financial-table th {
  padding: 12px 8px;
  text-align: left;
  font-weight: 600;
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  white-space: nowrap;
}

.financial-table td {
  padding: 10px 8px;
  border-bottom: 1px solid #e8ecef;
}

.expand-col { width: 40px; text-align: center; }
.line-col { width: 60px; text-align: center; }
.rsn-col { width: 120px; }
.pdli-num-col { width: 100px; }
.desc-col { width: 160px; }
.amount-col { width: 120px; text-align: right; }

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
.pdli-count { margin-left: 6px; color: #888; font-size: 0.75rem; font-weight: normal; }
.detail-rsn { padding-left: 8px; color: #666; }

.amount {
  font-family: 'SF Mono', 'Consolas', monospace;
  font-weight: 500;
  text-align: right;
}

.amount.primary { color: #e67e22; font-weight: 600; }
.amount.positive { color: #27ae60; }
.amount.negative { color: #e74c3c; }
</style>