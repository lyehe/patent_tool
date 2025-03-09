from patent_client._async import USApplication

apps = list()
async for app in USApplication.objects.filter(first_named_applicant="Google"):
  apps.append(app)

app = await USApplication.objects.aget("16123456")

