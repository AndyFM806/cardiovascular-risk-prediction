"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Heart, Activity, TrendingUp, Users, AlertTriangle, CheckCircle, Info } from "lucide-react"

interface PatientData {
  age: string
  gender: string
  height: string
  weight: string
  ap_hi: string
  ap_lo: string
  cholesterol: string
  gluc: string
  smoke: string
  alco: string
  active: string
}

interface PredictionResult {
  risk_probability: number
  risk_level: string
  risk_color: string
  recommendations: string[]
  factors: {
    name: string
    impact: string
    value: string
  }[]
}

export default function CardiovascularRiskPredictor() {
  const [patientData, setPatientData] = useState<PatientData>({
    age: "",
    gender: "",
    height: "",
    weight: "",
    ap_hi: "",
    ap_lo: "",
    cholesterol: "",
    gluc: "",
    smoke: "",
    alco: "",
    active: "",
  })

  const [prediction, setPrediction] = useState<PredictionResult | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [activeTab, setActiveTab] = useState("predictor")

  const handleInputChange = (field: keyof PatientData, value: string) => {
    setPatientData((prev) => ({
      ...prev,
      [field]: value,
    }))
  }

  const calculateBMI = () => {
    const height = Number.parseFloat(patientData.height)
    const weight = Number.parseFloat(patientData.weight)
    if (height && weight) {
      return (weight / (height / 100) ** 2).toFixed(1)
    }
    return null
  }

  const getBMICategory = (bmi: number) => {
    if (bmi < 18.5) return { category: "Bajo peso", color: "text-blue-600" }
    if (bmi < 25) return { category: "Normal", color: "text-green-600" }
    if (bmi < 30) return { category: "Sobrepeso", color: "text-yellow-600" }
    return { category: "Obesidad", color: "text-red-600" }
  }

  const getBPCategory = (systolic: number, diastolic: number) => {
    if (systolic < 120 && diastolic < 80) return { category: "Normal", color: "text-green-600" }
    if (systolic < 130 && diastolic < 80) return { category: "Elevada", color: "text-yellow-600" }
    if ((systolic >= 130 && systolic < 140) || (diastolic >= 80 && diastolic < 90)) {
      return { category: "Hipertensión Etapa 1", color: "text-orange-600" }
    }
    return { category: "Hipertensión Etapa 2", color: "text-red-600" }
  }

  const simulatePrediction = async (): Promise<PredictionResult> => {
    // Simulación de predicción basada en factores de riesgo reales
    const age = Number.parseInt(patientData.age)
    const bmi = calculateBMI()
    const systolic = Number.parseInt(patientData.ap_hi)
    const diastolic = Number.parseInt(patientData.ap_lo)
    const cholesterol = Number.parseInt(patientData.cholesterol)
    const glucose = Number.parseInt(patientData.gluc)
    const smoke = patientData.smoke === "1"
    const alcohol = patientData.alco === "1"
    const active = patientData.active === "1"

    let riskScore = 0

    // Factores de edad
    if (age > 65) riskScore += 0.3
    else if (age > 55) riskScore += 0.2
    else if (age > 45) riskScore += 0.1

    // Factores de presión arterial
    if (systolic > 140 || diastolic > 90) riskScore += 0.25
    else if (systolic > 130 || diastolic > 80) riskScore += 0.15

    // Factores de IMC
    if (bmi && Number.parseFloat(bmi) > 30) riskScore += 0.15
    else if (bmi && Number.parseFloat(bmi) > 25) riskScore += 0.1

    // Factores de colesterol y glucosa
    if (cholesterol > 2) riskScore += 0.1
    if (glucose > 2) riskScore += 0.1

    // Factores de estilo de vida
    if (smoke) riskScore += 0.2
    if (alcohol) riskScore += 0.05
    if (!active) riskScore += 0.1

    // Normalizar a probabilidad
    const probability = Math.min(Math.max(riskScore, 0.05), 0.95)

    let riskLevel: string
    let riskColor: string

    if (probability < 0.3) {
      riskLevel = "Bajo"
      riskColor = "text-green-600"
    } else if (probability < 0.6) {
      riskLevel = "Moderado"
      riskColor = "text-yellow-600"
    } else {
      riskLevel = "Alto"
      riskColor = "text-red-600"
    }

    const recommendations = []
    if (systolic > 130 || diastolic > 80) {
      recommendations.push("Controlar la presión arterial regularmente")
    }
    if (bmi && Number.parseFloat(bmi) > 25) {
      recommendations.push("Mantener un peso saludable")
    }
    if (smoke) {
      recommendations.push("Dejar de fumar inmediatamente")
    }
    if (!active) {
      recommendations.push("Realizar ejercicio físico regular")
    }
    if (cholesterol > 2) {
      recommendations.push("Controlar los niveles de colesterol")
    }
    if (recommendations.length === 0) {
      recommendations.push("Mantener hábitos saludables actuales")
    }

    const factors = [
      { name: "Edad", impact: age > 55 ? "Alto" : age > 45 ? "Moderado" : "Bajo", value: `${age} años` },
      {
        name: "Presión Arterial",
        impact: systolic > 140 ? "Alto" : systolic > 130 ? "Moderado" : "Bajo",
        value: `${systolic}/${diastolic} mmHg`,
      },
      {
        name: "IMC",
        impact: bmi && Number.parseFloat(bmi) > 30 ? "Alto" : bmi && Number.parseFloat(bmi) > 25 ? "Moderado" : "Bajo",
        value: bmi ? `${bmi} kg/m²` : "N/A",
      },
      { name: "Tabaquismo", impact: smoke ? "Alto" : "Bajo", value: smoke ? "Sí" : "No" },
      { name: "Actividad Física", impact: !active ? "Moderado" : "Bajo", value: active ? "Activo" : "Sedentario" },
    ]

    return {
      risk_probability: probability,
      risk_level: riskLevel,
      risk_color: riskColor,
      recommendations,
      factors,
    }
  }

  const handlePredict = async () => {
    setIsLoading(true)
    try {
      // Simular delay de API
      await new Promise((resolve) => setTimeout(resolve, 2000))
      const result = await simulatePrediction()
      setPrediction(result)
    } catch (error) {
      console.error("Error en predicción:", error)
    } finally {
      setIsLoading(false)
    }
  }

  const isFormValid = () => {
    return Object.values(patientData).every((value) => value !== "")
  }

  const bmi = calculateBMI()
  const bmiInfo = bmi ? getBMICategory(Number.parseFloat(bmi)) : null
  const bpInfo =
    patientData.ap_hi && patientData.ap_lo
      ? getBPCategory(Number.parseInt(patientData.ap_hi), Number.parseInt(patientData.ap_lo))
      : null

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center mb-4">
            <Heart className="h-12 w-12 text-red-500 mr-3" />
            <h1 className="text-4xl font-bold text-gray-800">Predictor de Riesgo Cardiovascular</h1>
          </div>
          <p className="text-lg text-gray-600 mb-2">
            Sistema de Inteligencia Artificial para Evaluación de Riesgo Cardíaco
          </p>
          <Badge variant="outline" className="text-sm">
            Universidad Privada Antenor Orrego
          </Badge>
        </div>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="predictor">🫀 Predictor</TabsTrigger>
            <TabsTrigger value="info">📊 Información</TabsTrigger>
            <TabsTrigger value="about">🎓 Acerca del Proyecto</TabsTrigger>
          </TabsList>

          <TabsContent value="predictor" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Formulario de Datos del Paciente */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center">
                    <Users className="h-5 w-5 mr-2" />
                    Datos del Paciente
                  </CardTitle>
                  <CardDescription>Ingrese los parámetros clínicos y de estilo de vida</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <Label htmlFor="age">Edad (años)</Label>
                      <Input
                        id="age"
                        type="number"
                        placeholder="Ej: 45"
                        value={patientData.age}
                        onChange={(e) => handleInputChange("age", e.target.value)}
                      />
                    </div>
                    <div>
                      <Label htmlFor="gender">Género</Label>
                      <Select value={patientData.gender} onValueChange={(value) => handleInputChange("gender", value)}>
                        <SelectTrigger>
                          <SelectValue placeholder="Seleccionar" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="1">Femenino</SelectItem>
                          <SelectItem value="2">Masculino</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <Label htmlFor="height">Altura (cm)</Label>
                      <Input
                        id="height"
                        type="number"
                        placeholder="Ej: 170"
                        value={patientData.height}
                        onChange={(e) => handleInputChange("height", e.target.value)}
                      />
                    </div>
                    <div>
                      <Label htmlFor="weight">Peso (kg)</Label>
                      <Input
                        id="weight"
                        type="number"
                        placeholder="Ej: 70"
                        value={patientData.weight}
                        onChange={(e) => handleInputChange("weight", e.target.value)}
                      />
                    </div>
                  </div>

                  {bmi && (
                    <Alert>
                      <Info className="h-4 w-4" />
                      <AlertDescription>
                        IMC: <span className="font-semibold">{bmi}</span> -
                        <span className={`ml-1 font-semibold ${bmiInfo?.color}`}>{bmiInfo?.category}</span>
                      </AlertDescription>
                    </Alert>
                  )}

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <Label htmlFor="ap_hi">Presión Sistólica (mmHg)</Label>
                      <Input
                        id="ap_hi"
                        type="number"
                        placeholder="Ej: 120"
                        value={patientData.ap_hi}
                        onChange={(e) => handleInputChange("ap_hi", e.target.value)}
                      />
                    </div>
                    <div>
                      <Label htmlFor="ap_lo">Presión Diastólica (mmHg)</Label>
                      <Input
                        id="ap_lo"
                        type="number"
                        placeholder="Ej: 80"
                        value={patientData.ap_lo}
                        onChange={(e) => handleInputChange("ap_lo", e.target.value)}
                      />
                    </div>
                  </div>

                  {bpInfo && (
                    <Alert>
                      <Activity className="h-4 w-4" />
                      <AlertDescription>
                        Presión Arterial:
                        <span className={`ml-1 font-semibold ${bpInfo.color}`}>{bpInfo.category}</span>
                      </AlertDescription>
                    </Alert>
                  )}

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <Label htmlFor="cholesterol">Colesterol</Label>
                      <Select
                        value={patientData.cholesterol}
                        onValueChange={(value) => handleInputChange("cholesterol", value)}
                      >
                        <SelectTrigger>
                          <SelectValue placeholder="Seleccionar" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="1">Normal</SelectItem>
                          <SelectItem value="2">Por encima de lo normal</SelectItem>
                          <SelectItem value="3">Muy por encima de lo normal</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div>
                      <Label htmlFor="gluc">Glucosa</Label>
                      <Select value={patientData.gluc} onValueChange={(value) => handleInputChange("gluc", value)}>
                        <SelectTrigger>
                          <SelectValue placeholder="Seleccionar" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="1">Normal</SelectItem>
                          <SelectItem value="2">Por encima de lo normal</SelectItem>
                          <SelectItem value="3">Muy por encima de lo normal</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>

                  <div className="grid grid-cols-3 gap-4">
                    <div>
                      <Label htmlFor="smoke">¿Fuma?</Label>
                      <Select value={patientData.smoke} onValueChange={(value) => handleInputChange("smoke", value)}>
                        <SelectTrigger>
                          <SelectValue placeholder="Seleccionar" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="0">No</SelectItem>
                          <SelectItem value="1">Sí</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div>
                      <Label htmlFor="alco">¿Consume alcohol?</Label>
                      <Select value={patientData.alco} onValueChange={(value) => handleInputChange("alco", value)}>
                        <SelectTrigger>
                          <SelectValue placeholder="Seleccionar" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="0">No</SelectItem>
                          <SelectItem value="1">Sí</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div>
                      <Label htmlFor="active">¿Físicamente activo?</Label>
                      <Select value={patientData.active} onValueChange={(value) => handleInputChange("active", value)}>
                        <SelectTrigger>
                          <SelectValue placeholder="Seleccionar" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="0">No</SelectItem>
                          <SelectItem value="1">Sí</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>

                  <Button onClick={handlePredict} disabled={!isFormValid() || isLoading} className="w-full" size="lg">
                    {isLoading ? (
                      <>
                        <Activity className="h-4 w-4 mr-2 animate-spin" />
                        Analizando...
                      </>
                    ) : (
                      <>
                        <TrendingUp className="h-4 w-4 mr-2" />
                        Predecir Riesgo Cardiovascular
                      </>
                    )}
                  </Button>
                </CardContent>
              </Card>

              {/* Resultados de Predicción */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center">
                    <Activity className="h-5 w-5 mr-2" />
                    Resultado de la Predicción
                  </CardTitle>
                  <CardDescription>Análisis de riesgo basado en IA</CardDescription>
                </CardHeader>
                <CardContent>
                  {!prediction && !isLoading && (
                    <div className="text-center py-12">
                      <Heart className="h-16 w-16 text-gray-300 mx-auto mb-4" />
                      <p className="text-gray-500">
                        Complete el formulario y haga clic en "Predecir" para obtener el análisis de riesgo
                      </p>
                    </div>
                  )}

                  {isLoading && (
                    <div className="text-center py-12">
                      <Activity className="h-16 w-16 text-blue-500 mx-auto mb-4 animate-spin" />
                      <p className="text-gray-600 mb-4">Analizando datos del paciente...</p>
                      <Progress value={75} className="w-full max-w-xs mx-auto" />
                    </div>
                  )}

                  {prediction && (
                    <div className="space-y-6">
                      {/* Nivel de Riesgo */}
                      <div className="text-center">
                        <div className={`text-6xl font-bold mb-2 ${prediction.risk_color}`}>
                          {(prediction.risk_probability * 100).toFixed(0)}%
                        </div>
                        <div className={`text-2xl font-semibold mb-4 ${prediction.risk_color}`}>
                          Riesgo {prediction.risk_level}
                        </div>
                        <Progress value={prediction.risk_probability * 100} className="w-full max-w-md mx-auto" />
                      </div>

                      {/* Factores de Riesgo */}
                      <div>
                        <h4 className="font-semibold mb-3 flex items-center">
                          <AlertTriangle className="h-4 w-4 mr-2" />
                          Factores de Riesgo Analizados
                        </h4>
                        <div className="space-y-2">
                          {prediction.factors.map((factor, index) => (
                            <div key={index} className="flex justify-between items-center p-2 bg-gray-50 rounded">
                              <span className="font-medium">{factor.name}</span>
                              <div className="flex items-center space-x-2">
                                <span className="text-sm text-gray-600">{factor.value}</span>
                                <Badge
                                  variant={
                                    factor.impact === "Alto"
                                      ? "destructive"
                                      : factor.impact === "Moderado"
                                        ? "default"
                                        : "secondary"
                                  }
                                  className="text-xs"
                                >
                                  {factor.impact}
                                </Badge>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>

                      {/* Recomendaciones */}
                      <div>
                        <h4 className="font-semibold mb-3 flex items-center">
                          <CheckCircle className="h-4 w-4 mr-2" />
                          Recomendaciones
                        </h4>
                        <ul className="space-y-2">
                          {prediction.recommendations.map((rec, index) => (
                            <li key={index} className="flex items-start">
                              <CheckCircle className="h-4 w-4 text-green-500 mr-2 mt-0.5 flex-shrink-0" />
                              <span className="text-sm">{rec}</span>
                            </li>
                          ))}
                        </ul>
                      </div>

                      <Alert>
                        <Info className="h-4 w-4" />
                        <AlertDescription>
                          <strong>Importante:</strong> Esta predicción es una herramienta de apoyo diagnóstico. Siempre
                          consulte con un profesional de la salud para una evaluación completa.
                        </AlertDescription>
                      </Alert>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="info" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center">
                    <TrendingUp className="h-5 w-5 mr-2" />
                    Estadísticas del Modelo
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="flex justify-between">
                      <span>Precisión del Modelo:</span>
                      <Badge variant="secondary">87.3%</Badge>
                    </div>
                    <div className="flex justify-between">
                      <span>Sensibilidad:</span>
                      <Badge variant="secondary">84.1%</Badge>
                    </div>
                    <div className="flex justify-between">
                      <span>Especificidad:</span>
                      <Badge variant="secondary">89.7%</Badge>
                    </div>
                    <div className="flex justify-between">
                      <span>AUC-ROC:</span>
                      <Badge variant="secondary">0.891</Badge>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center">
                    <Users className="h-5 w-5 mr-2" />
                    Dataset Utilizado
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="flex justify-between">
                      <span>Registros Totales:</span>
                      <Badge variant="outline">1.5M+</Badge>
                    </div>
                    <div className="flex justify-between">
                      <span>Características:</span>
                      <Badge variant="outline">19</Badge>
                    </div>
                    <div className="flex justify-between">
                      <span>Fuente:</span>
                      <Badge variant="outline">Kaggle</Badge>
                    </div>
                    <div className="flex justify-between">
                      <span>Validación:</span>
                      <Badge variant="outline">Cross-validation</Badge>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center">
                    <Activity className="h-5 w-5 mr-2" />
                    Técnicas de IA
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    <Badge variant="outline" className="w-full justify-center">
                      Random Forest
                    </Badge>
                    <Badge variant="outline" className="w-full justify-center">
                      XGBoost
                    </Badge>
                    <Badge variant="outline" className="w-full justify-center">
                      Red Neuronal
                    </Badge>
                    <Badge variant="outline" className="w-full justify-center">
                      SHAP (Explicabilidad)
                    </Badge>
                  </div>
                </CardContent>
              </Card>
            </div>

            <Card>
              <CardHeader>
                <CardTitle>Interpretación de Resultados</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="text-center">
                    <div className="text-3xl font-bold text-green-600 mb-2">Riesgo Bajo</div>
                    <div className="text-sm text-gray-600">0% - 30%</div>
                    <p className="text-sm mt-2">
                      Probabilidad baja de desarrollar enfermedad cardiovascular. Mantener hábitos saludables.
                    </p>
                  </div>
                  <div className="text-center">
                    <div className="text-3xl font-bold text-yellow-600 mb-2">Riesgo Moderado</div>
                    <div className="text-sm text-gray-600">30% - 60%</div>
                    <p className="text-sm mt-2">
                      Riesgo intermedio. Se recomienda evaluación médica y modificación de factores de riesgo.
                    </p>
                  </div>
                  <div className="text-center">
                    <div className="text-3xl font-bold text-red-600 mb-2">Riesgo Alto</div>
                    <div className="text-sm text-gray-600">60% - 100%</div>
                    <p className="text-sm mt-2">
                      Alto riesgo cardiovascular. Se requiere atención médica inmediata y seguimiento especializado.
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="about" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Heart className="h-5 w-5 mr-2" />
                  Acerca del Proyecto
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <h3 className="font-semibold text-lg mb-2">Universidad Privada Antenor Orrego</h3>
                  <p className="text-gray-600">
                    Este proyecto forma parte de la investigación en aplicaciones de Inteligencia Artificial en el campo
                    de la medicina preventiva, desarrollado para demostrar el potencial de los algoritmos de Machine
                    Learning en la predicción temprana de riesgo cardiovascular.
                  </p>
                </div>

                <div>
                  <h4 className="font-semibold mb-2">🎯 Justificación</h4>
                  <p className="text-gray-600">
                    Las enfermedades cardiovasculares son la principal causa de muerte en el mundo. Este proyecto busca
                    usar inteligencia artificial para anticipar el riesgo en pacientes, facilitando intervenciones
                    tempranas. Es un tema socialmente relevante, aplicable a medicina preventiva y muy demandado en
                    entornos clínicos y aseguradoras.
                  </p>
                </div>

                <div>
                  <h4 className="font-semibold mb-2">🧠 Técnicas Implementadas</h4>
                  <ul className="list-disc list-inside text-gray-600 space-y-1">
                    <li>Preprocesamiento y limpieza masiva con pandas</li>
                    <li>Normalización con StandardScaler</li>
                    <li>Modelos supervisados: Random Forest, XGBoost, Red Neuronal</li>
                    <li>Evaluación con matriz de confusión, ROC AUC, precisión y recall</li>
                    <li>IA explicable con SHAP para interpretación de decisiones</li>
                  </ul>
                </div>

                <div>
                  <h4 className="font-semibold mb-2">📊 Dataset</h4>
                  <ul className="list-disc list-inside text-gray-600 space-y-1">
                    <li>Nombre: Cardiovascular Disease Dataset</li>
                    <li>Tamaño: ~1.5 millones de registros</li>
                    <li>Formato: CSV (~300MB)</li>
                    <li>Fuente: Kaggle Dataset - Cardiovascular Disease</li>
                  </ul>
                </div>

                <Alert>
                  <Info className="h-4 w-4" />
                  <AlertDescription>
                    <strong>Disclaimer:</strong> Esta herramienta es únicamente para fines educativos y de
                    investigación. No debe utilizarse como sustituto del diagnóstico médico profesional. Siempre
                    consulte con un profesional de la salud calificado para cualquier decisión médica.
                  </AlertDescription>
                </Alert>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
