<template>
  <div class="portfolio-page" :style="{ marginLeft: sidebarMarginLeft }">
    <PortfolioView @case-clicked="handleCaseSelection" />
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import PortfolioView from '@/components/dashboard/tabs/PortfolioView.vue'

const props = defineProps({
  isSidebarCollapsed: {
    type: Boolean,
    default: false
  }
})

const router = useRouter()

// Match the same margin calculation as MainDashboardLayout
const sidebarMarginLeft = computed(() => {
  const expandedSidebarWidth = "250px"
  const collapsedSidebarWidth = "70px"
  return props.isSidebarCollapsed ? collapsedSidebarWidth : expandedSidebarWidth
})

function handleCaseSelection(caseId) {
  console.log('[PortfolioPage] Navigating to case:', caseId)
  router.push({ name: 'Dashboard', params: { caseId } })
}
</script>

<style scoped>
.portfolio-page {
  flex: 1;
  overflow: auto;
  background: #f8fafc;
  min-height: 100vh;
  transition: margin-left 0.3s ease;
}
</style>