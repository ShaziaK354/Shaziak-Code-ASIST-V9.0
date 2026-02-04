<template>
  <div class="dashboard-page">
    <MainDashboardLayout :is-sidebar-collapsed="isSidebarCollapsed" />
  </div>
</template>

<script setup>
import { watch, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import MainDashboardLayout from '@/components/layout/MainDashboardLayout.vue'
import { useCaseStore } from '@/stores/caseStore'

const props = defineProps({
  isSidebarCollapsed: {
    type: Boolean,
    default: false
  }
})

const route = useRoute()
const caseStore = useCaseStore()

// Watch route changes to update active case
watch(() => route.params.caseId, (newCaseId) => {
  if (newCaseId) {
    console.log('[DashboardPage] Route caseId changed:', newCaseId)
    caseStore.setActiveCase(newCaseId)
  }
}, { immediate: true })

// On mount, set default case if none in route
onMounted(() => {
  const caseId = route.params.caseId || 'SR-P-NAV'
  console.log('[DashboardPage] Mounted with caseId:', caseId)
  if (!caseStore.activeCaseDetails) {
    caseStore.setActiveCase(caseId)
  }
})
</script>

<style scoped>
.dashboard-page {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 100vh;
}
</style>